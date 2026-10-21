"""Single-pass GT task policy over the evaluated physical tool registry.

GT selects task endpoints, target identity and authored answers. Navigation is
the harness PointGoal A* handler over official occupancy; grasping still needs
fresh SAM3, exact same-frame confirmation, GraspGenX, cuRobo and physical hold.
There is no episode restart, alternate grasp backend or direct controller path.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tarsphere.harness.agent import agent_observation_from_runtime
from tarsphere.harness.core.types import ToolCall, ToolResult
from tarsphere.harness.harness import (
    AgentHarnessConfig,
)
from tarsphere.harness.certification.grasp_skill_agent import (
    GRASP_CERTIFICATION_CAMERA_TILT_DEG,
    PHYSICAL_TASK_NAVIGATION_TOLERANCE_M,
    PHYSICAL_TASK_HEADING_TOLERANCE_DEG,
)
from tarsphere.harness.planner.planner import world_segment_cell_intervals
from tarsphere.bench.certification.scene import (
    load_episode_navigation_grid,
    load_episode_occupancy,
)
from tarsphere.bench.production_layout import FETCH_STANCE_DISTANCE_BAND_M
from tarsphere.contracts.task_profile import CapabilitySlice
from tarsphere.infra.control.physical_contracts import (
    LOCOMOTION_MAX_TURN_DEG,
    LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
    LOCOMOTION_MAX_LATERAL_DISTANCE_M,
    LOCOMOTION_MIN_STEP_DISTANCE_M,
    LOCOMOTION_MIN_TURN_DEG,
    G1_RELATIVE_POSE_TRANSLATION_TOLERANCE_M,
    NAVIGATION_SEGMENT_MAX_TURN_RAD,
    HANDOVER_MIN_DISTANCE_M,
    HANDOVER_MAX_DISTANCE_M,
    NAVIGATION_MIN_HEADING_PROGRESS_RAD,
    NAVIGATION_MIN_XY_PROGRESS_M,
    NAVIGATION_STUCK_TERMINATION_DECISIONS,
)
from tarsphere.infra.manipulation.handover import horizontal_distance_m


# Leave room for measured execution error inside the existing 0.12 m arrival
# contract. The ordinary 0.50 m goal region cannot certify a grasp stance.
REFERENCE_NAVIGATION_GOAL_RADIUS_M = PHYSICAL_TASK_NAVIGATION_TOLERANCE_M / 2
REFERENCE_FINAL_APPROACH_STEP_M = (
    min(
        LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
        LOCOMOTION_MAX_LATERAL_DISTANCE_M,
    )
    - REFERENCE_NAVIGATION_GOAL_RADIUS_M
)
# Use the same physically certified bounded translation as the final approach.
# Repeated 0.10 m loaded retreats fall into a controller dead zone on several
# scenes: measured motion can be opposite to the request and a second pulse can
# drop the object.  A single approximately 0.44 m departure is already proven
# across the unchanged standard locomotion/relative-pose controller stack.
REFERENCE_LOADED_DEPARTURE_STEP_M = REFERENCE_FINAL_APPROACH_STEP_M
# Once the body has cleared all but 0.10 m of the dedicated 0.44 m retreat,
# ordinary return A* can safely take over.  Requiring centimetre-scale loaded
# correction at the worktop adds no clearance and can trap the gait deadband.
REFERENCE_LOADED_DEPARTURE_GOAL_RADIUS_M = LOCOMOTION_MIN_STEP_DISTANCE_M
# Cross-scene A* only needs to reach the hand-off neighbourhood before the
# separately checked heading-preserving final approach.  Chasing an ordinary
# intermediate checkpoint to the final 0.12 m task radius can turn a
# centimetre-scale gait overshoot into an oscillation without improving the
# eventual grasp stance proof.
REFERENCE_INTERMEDIATE_CHECKPOINT_RADIUS_M = 2 * LOCOMOTION_MIN_STEP_DISTANCE_M
# Snap plus controller deadband must fit inside half the conservative band.
REFERENCE_GRASP_GOAL_RADIUS_M = (
    0.01875  # Local motion resolution; not the reach-envelope width.
)
# The direct grasp certification starts at authored B. Reproduce that proven
# launch pose to the same local-resolution radius before SAM/cuRobo; a 2.8 cm
# drift has already changed an otherwise certified can from reachable to IK
# failure. This remains an endpoint tolerance, not a wider B authoring region.
REFERENCE_PRECHECK_STANCE_TOLERANCE_M = G1_RELATIVE_POSE_TRANSLATION_TOLERANCE_M
# Direct grasp prechecks face the target before SAM/curobo. Leave the authored
# B-region unchanged, but reserve half of its public heading tolerance for gait
# drift so a target near the edge is re-centred by XY before grasp execution.
REFERENCE_GRASP_HEADING_TOLERANCE_DEG = PHYSICAL_TASK_HEADING_TOLERANCE_DEG / 2
# A fresh GraspGenX sample can be planning-feasible yet miss the live Dex3
# closure after controller tracking. Keep the certification run finite while
# allowing two recovery attempts; every attempt still traverses the shared
# public perception, planning, execution, and measured-retention path.
REFERENCE_GRASP_EXECUTION_LIMIT = 3
# One gait pulse can move opposite its requested endpoint while still upright.
# Replan once from the measured pose, then stop rather than repeating an
# unstable loaded translation indefinitely.
REFERENCE_LOADED_DEPARTURE_STALL_LIMIT = 2


@dataclass(frozen=True)
class MeasuredGraspStance:
    """Current body/target geometry, never an earlier arrival latch."""

    body_clear: bool  # Actual body centre is clear in the official inflated map.
    stance_error_m: float  # Horizontal displacement from the original authored B.
    target_distance_m: float  # Body-to-live-target AABB centre horizontal distance.
    heading_error_deg: float  # Absolute yaw error toward that live centre.

    @property
    def position_valid(self) -> bool:
        return (
            self.body_clear
            # The 0.12 m B region is an authoring-validity envelope.  A
            # physical reference must reproduce the direct-precheck stance,
            # rather than treating every point in that envelope as an
            # interchangeable cuRobo launch pose.
            and self.stance_error_m <= REFERENCE_PRECHECK_STANCE_TOLERANCE_M
            and FETCH_STANCE_DISTANCE_BAND_M[0]
            <= self.target_distance_m
            <= FETCH_STANCE_DISTANCE_BAND_M[1]
        )

    @property
    def valid(self) -> bool:
        return (
            self.position_valid
            and self.heading_error_deg <= REFERENCE_GRASP_HEADING_TOLERANCE_DEG
        )


def grasp_approach_checkpoint(path: list[Any]) -> tuple[float, float]:
    """Hand off from ordinary A* just before the bounded local approach.

    The ordinary PointGoal arrival band belongs outside the maximum local gait
    step.  Otherwise a controller can land just beyond the intermediate
    checkpoint and chase centimetres with repeated whole-body steps.  The
    remaining table-side segment is still executed as separately observed,
    bounded motions and ends at the unchanged authored stance.
    """
    remaining = (
        min(LOCOMOTION_MAX_BACKWARD_DISTANCE_M, LOCOMOTION_MAX_LATERAL_DISTANCE_M)
        + PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
    )
    for start, end in reversed(list(zip(path, path[1:]))):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end - start))
        if length >= remaining:
            return tuple(end + (start - end) * (remaining / length))
        remaining -= length
    return tuple(path[0])


def task_departure_checkpoint(path: list[Any]) -> tuple[float, float]:
    """Return one bounded, path-preserving checkpoint away from a task site."""
    if len(path) < 2:
        raise ValueError("Task departure needs a non-empty A* segment.")
    distance = min(
        LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
        LOCOMOTION_MAX_LATERAL_DISTANCE_M,
    )
    travelled = 0.0
    for segment_start, segment_end in zip(path, path[1:]):
        segment_start = np.asarray(segment_start, dtype=float)
        segment_end = np.asarray(segment_end, dtype=float)
        length = float(np.linalg.norm(segment_end - segment_start))
        if travelled + length >= distance:
            fraction = (distance - travelled) / length
            return tuple(segment_start + fraction * (segment_end - segment_start))
        travelled += length
    return tuple(np.asarray(path[-1], dtype=float))


def task_approach_checkpoint(grid: Any, path: list[Any]) -> tuple[float, float]:
    """Find the last roomy point on the actual outbound A* path for facing B.

    The remaining path is traversed by bounded heading-preserving PointGoals.
    Do not rotate next to the inspected furniture just because B is reachable.
    """
    for start, end in reversed(list(zip(path, path[1:]))):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        count = max(1, int(math.ceil(np.linalg.norm(end - start) / grid.resolution)))
        for fraction in np.linspace(0.0, 1.0, count + 1):
            point = end + fraction * (start - end)
            if grid.can_turn_in_place(point):
                return tuple(point)
    # A narrow but traversable scene must not acquire a larger hard footprint.
    # Retain the existing bounded approach rule; physical evidence still owns
    # success, and this fallback never claims certified turning clearance.
    return grasp_approach_checkpoint(path)


def grasp_departure_checkpoint(
    grid: Any, stance: Any, yaw: float, approach: Any
) -> tuple[float, float]:
    """Prefer a clear retreat away from the worktop, not an outbound retrace.

    Approaching a worktop around chairs does not make that same lateral path
    a sensible loaded departure. Select a straight backward endpoint only
    when its entire segment is free in the unchanged official body mask.
    Otherwise retain the existing A* approach checkpoint. Execution still
    replans from measured pose through the shared PointGoal handler.
    """
    start = np.asarray(stance, dtype=float)
    distance = min(
        LOCOMOTION_MAX_BACKWARD_DISTANCE_M, LOCOMOTION_MAX_LATERAL_DISTANCE_M
    )
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    retreat = start - distance * forward
    if grid.is_navigable(retreat) and grid._planner().is_line_collision_free(
        start, retreat
    ):
        return tuple(retreat)
    # The back ray can be blocked by an aisle boundary. Leave sideways or
    # diagonally backward before rejoining A*, never inch toward the worktop
    # merely to reach a previously authored approach point. Rank equivalent
    # clear exits toward the intended return direction.
    candidates = [
        start + distance * (-math.cos(angle) * forward + sign * math.sin(angle) * left)
        for angle in (math.pi / 6, math.pi / 3, math.pi / 2)
        for sign in (-1, 1)
    ]
    for point in sorted(candidates, key=lambda p: np.linalg.norm(p - approach)):
        if grid.is_navigable(point) and grid._planner().is_line_collision_free(
            start, point
        ):
            return tuple(point)
    return tuple(approach)


def require_table_side_translation(result: ToolResult) -> None:
    """Reject a handler-produced turn during local grasp approach/departure."""
    if (
        result.action is not None
        and result.action.get("action") == "move_body"
        and abs(float(result.action["delta"][2])) > 1e-9
    ):
        raise RuntimeError("Grasp approach/departure cannot execute a table-side turn.")


def live_navigation_humans(
    episode: Any,
    world_snapshot: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """Return every authored Human's current measured XY for PointGoal A*.

    This private reference input is intentionally rebuilt from the same world
    snapshot the evaluator records before every decision.  Missing or malformed
    actor poses are fatal: silently reverting to authored spawn coordinates
    would make a moving or drifted body disappear from collision planning.
    """

    expected = tuple(str(human.id) for human in episode.humans)
    if not expected:
        return {}
    humans = world_snapshot.get("humans")
    if not isinstance(humans, dict):
        raise RuntimeError(
            "Reference PointGoal requires the current world snapshot for every Human."
        )
    positions: dict[str, tuple[float, float]] = {}
    for human_id in expected:
        state = humans.get(human_id)
        raw_position = state.get("position") if isinstance(state, dict) else None
        try:
            position = np.asarray(raw_position, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Reference PointGoal received an invalid live pose for Human {human_id!r}."
            ) from exc
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            raise RuntimeError(
                f"Reference PointGoal received no finite live XY for Human {human_id!r}."
            )
        positions[human_id] = (float(position[0]), float(position[1]))
    return positions


def reference_navigation(episode: Any) -> tuple[Any, dict[str, Any]]:
    """Preflight both legs with exactly the runtime map and PointGoal snapping.

    Authored route length is not the length of a fresh runtime A* solution.
    Persist both, including the resolved cell and its metric snap displacement.
    """
    if episode.evaluation.task_profile.capability_slice is not CapabilitySlice.BASE:
        raise ValueError(
            "Harness reference currently admits frozen base episodes only."
        )
    if any(not h.static for h in episode.humans):
        raise ValueError("Static-map reference requires authored static Humans.")
    route = episode.world_knowledge["episode_route"]
    source = Path(load_episode_occupancy(episode).occupancy_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != route["occupancy_sha256"]:
        raise ValueError("Reference occupancy hash differs from the frozen route.")
    grid = load_episode_navigation_grid(episode)
    start = np.asarray(episode.robot.start_position[:2], dtype=float)
    legs = {}
    for name in ("navigate_to_target", "transport_to_destination"):
        goal = np.asarray(route[name]["points"][-1]["position_xy"], dtype=float)
        points, snapped, reason = grid.astar_to_goal_region(
            start,
            goal,
            lateral_tolerance_m=0.25,
            search_radius_m=REFERENCE_NAVIGATION_GOAL_RADIUS_M,
        )
        if points is None:
            raise ValueError(
                f"{episode.episode_id}/{name}: runtime A* rejected route: {reason}"
            )
        legs[name] = {
            "requested_goal_xy": goal.tolist(),
            "resolved_goal_xy": snapped.tolist(),
            "snap_distance_m": float(np.linalg.norm(snapped - goal)),
            "path_xy": [np.asarray(p).tolist() for p in points],
            "length_m": sum(
                float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])
            ),
        }
        start = snapped
    return grid, {
        "occupancy_sha256": digest,
        "authored_reference_length_m": route["optimal_navigation_length_m"],
        "initial_runtime_astar_length_m": sum(leg["length_m"] for leg in legs.values()),
        "navigation_legs": legs,
    }


class HarnessReferenceAgent:
    """Execute a frozen base task once and archive its measured observations."""

    name = "harness_reference"
    certification_only = True

    def grasp_query(self) -> str:
        """Use the complete public visual referent from the task request."""
        return self.specific_grasp_query()

    def specific_grasp_query(self) -> str:
        """Return the authored visual referent without its support location."""
        import re

        name = self.target.get("name")
        if not name:
            return str(
                self.target.get("category") or self.target.get("object_id") or ""
            )
        return re.split(
            r"\s+(?:placed|on|at|beside|near)\b",
            name,
            maxsplit=1,
        )[0]

    def __init__(
        self, *, agent_harness_config: AgentHarnessConfig, tools: Any, output_dir: Path
    ) -> None:
        self.agent_harness_config = agent_harness_config
        self.tools = tools
        self.output_root = Path(output_dir)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.phase = "dispatch"
        self.completed: set[str] = set()
        self.opened: set[str] = set()
        self.feedback: dict[str, Any] = {}
        self.physical_grasp_feedback: dict[str, Any] = {}
        self.grasp_reachability_feedback: dict[str, Any] = {}
        self.active_grasp_stance: tuple[float, float] | None = None
        self.world_snapshot: dict[str, Any] = {}
        self.wait_steps = 0
        self.best_heading_error_deg = math.inf
        self.heading_stall_decisions = 0
        self.last_call = ""
        self.navigation_static_contact_streak = 0
        self.navigation_recovery_attempts = 0
        self.loaded_departure_stall_decisions = 0
        self.grasp_executions = 0
        self.grasp_stance_recovery_attempts = 0
        self.completion_attempts = 0
        self._attempt_started = False
        self.inquiry = False

    def configure_episode(self, episode: Any) -> None:
        self.grid, navigation = reference_navigation(episode)
        self.episode = episode
        milestones = tuple(episode.evaluation.milestones)
        if len(milestones) != 5:
            raise ValueError("Reference base must have five task stages.")
        # Roles distinguish source discovery from return visibility even when
        # both stages use the same primitive evaluator.
        self.stages = dict(
            zip(
                ("plan", "discovery", "onsite", "recipient", "completion"),
                milestones,
            )
        )
        self.fetch = self.stages["completion"].outcome.evaluator == "handover_accepted"
        self.inquiry = self.stages["onsite"].outcome.evaluator == "information_received"
        world = episode.world_knowledge
        route = world["episode_route"]
        self.target_stance = tuple(
            route["navigate_to_target"]["points"][-1]["position_xy"]
        )
        self.completion_stance = tuple(
            route["transport_to_destination"]["points"][-1]["position_xy"]
        )
        route_departure_stance = task_departure_checkpoint(
            navigation["navigation_legs"]["transport_to_destination"]["path_xy"],
        )
        if self.inquiry:
            truth = self.stages["onsite"].outcome.ground_truth
            self.source = next(h for h in episode.humans if h.id == truth["human_id"])
            self.inquiry_intent = next(
                intent
                for intent in episode.interaction.intents
                if intent.intent_id == truth["intent_id"]
            )
        else:
            observed = self.stages["discovery"].outcome.ground_truth["object_ids"]
            if len(observed) != 1:
                raise ValueError("Base reference requires one exact visual target.")
            self.target = next(
                o for o in world["objects"] if o["object_id"] == observed[0]
            )
        # At a close inspection/inquiry stance the body faces the observed
        # object or Human. Prefer a collision-checked straight retreat along
        # that heading before A* is allowed to turn the body. This uses the
        # same generic worktop-side rule as fetch departure and avoids a
        # diagonal strafe into nearby furniture. Fall back to the reviewed
        # return-path prefix when the backward segment is not clear.
        task_target_xy = np.asarray(
            self.source.start_position[:2]
            if self.inquiry
            else self.target["position_xyz"][:2],
            dtype=float,
        )
        task_yaw = math.atan2(
            task_target_xy[1] - self.target_stance[1],
            task_target_xy[0] - self.target_stance[0],
        )
        self.task_departure_stance = grasp_departure_checkpoint(
            self.grid,
            self.target_stance,
            task_yaw,
            route_departure_stance,
        )
        if not self.grid.is_navigable(np.asarray(self.task_departure_stance)):
            raise ValueError("Task-site departure checkpoint is not body-clear.")
        self.task_yaw = task_yaw
        self.task_approach_stance = (
            task_approach_checkpoint(
                self.grid,
                navigation["navigation_legs"]["navigate_to_target"]["path_xy"],
            )
            if not self.fetch
            else None
        )
        if self.fetch:
            # Turn only at a body-clear point on the actual outbound A* route.
            # The remaining path is executed as bounded, heading-preserving
            # translations, reproducing the physically certified precheck
            # stance without attempting an in-place turn beside furniture.
            route_approach_stance = task_approach_checkpoint(
                self.grid,
                navigation["navigation_legs"]["navigate_to_target"]["path_xy"],
            )
            self.grasp_yaw = math.atan2(
                self.target["position_xyz"][1] - self.target_stance[1],
                self.target["position_xyz"][0] - self.target_stance[0],
            )
            self.departure_stance = grasp_departure_checkpoint(
                self.grid,
                self.target_stance,
                self.grasp_yaw,
                route_approach_stance,
            )
            # Use one shared table-side buffer for both directions.  When the
            # checked backward ray is clear this makes the final approach a
            # straight forward motion and the loaded exit its exact reverse,
            # instead of forcing the gait to strafe around a soft-cost A*
            # corner while it must keep facing the target.
            self.approach_stance = self.departure_stance
            if not self.grid.is_navigable(np.asarray(self.approach_stance)):
                raise ValueError("Pregrasp turn checkpoint is not body-clear.")
        terminal = self.stages["completion"]
        self.recipient_id = terminal.outcome.ground_truth["human_id"]
        self.recipient = next(h for h in episode.humans if h.id == self.recipient_id)
        self.completion_approach_stance = task_approach_checkpoint(
            self.grid,
            navigation["navigation_legs"]["transport_to_destination"]["path_xy"],
        )
        self.completion_yaw = math.atan2(
            self.recipient.start_position[1] - self.completion_stance[1],
            self.recipient.start_position[0] - self.completion_stance[0],
        )
        self.output = self.output_root / episode.episode_id
        # One folder is one attempt. Never overwrite a failed attempt to turn
        # a later success into a purported first-pass optimal reference.
        self.output.mkdir(parents=True, exist_ok=False)
        self._attempt_started = True
        navigation_handler = self.tools.handler("navigate_to_pointgoal")
        if navigation_handler is not None:
            navigation_handler.reset()
            navigation_handler.configure_goal_region(
                search_radius_m=REFERENCE_NAVIGATION_GOAL_RADIUS_M,
            )
        self._save(
            "contract.json",
            {
                "episode_id": episode.episode_id,
                "episode_payload_sha256": hashlib.sha256(
                    episode.model_dump_json().encode()
                ).hexdigest(),
                "agent_harness_config": self.agent_harness_config.as_dict(),
                "reference_scope": "GT-guided fixed-endpoint A* reference; not globally optimal manipulation or duration",
                **navigation,
                "target_stance_xy": self.target_stance,
                "completion_stance_xy": self.completion_stance,
                "navigation_goal_radius_m": REFERENCE_NAVIGATION_GOAL_RADIUS_M,
                "grasp_goal_radius_m": REFERENCE_GRASP_GOAL_RADIUS_M
                if self.fetch
                else None,
                "pregrasp_turn_checkpoint_xy": self.approach_stance
                if self.fetch
                else None,
                "loaded_departure_checkpoint_xy": self.departure_stance
                if self.fetch
                else None,
                "task_departure_checkpoint_xy": self.task_departure_stance,
                "task_approach_checkpoint_xy": self.task_approach_stance,
                "completion_approach_checkpoint_xy": self.completion_approach_stance,
                **(
                    {
                        "source_human_id": self.source.id,
                        "intent_id": self.inquiry_intent.intent_id,
                    }
                    if self.inquiry
                    else {"target_object_id": self.target["object_id"]}
                ),
                "recipient_id": self.recipient_id,
                "attempt_limit": 1,
                "grasp_execution_limit": REFERENCE_GRASP_EXECUTION_LIMIT,
                "step_accounting": "physics_steps",
                "max_decision_steps": None,
                "max_physics_steps": None,
            },
        )

    def configure_episode_humans(self, humans: list[Any]) -> None:
        if {h.id for h in humans} != {h.id for h in self.episode.humans}:
            raise ValueError("Reference cast differs from its frozen episode.")

    def update_reference_state(
        self, *, snapshot: dict[str, Any], opened: tuple[str, ...], completed: set[str]
    ) -> None:
        """Runner-private task progress; this hook is never sent to a model."""
        self.world_snapshot = snapshot
        self.opened.update(opened)
        self.completed = set(completed)

    def _save(self, name: str, payload: Any) -> None:
        def numeric(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(
                f"Unsupported reference evidence type: {type(value).__name__}"
            )

        (self.output / name).write_text(
            json.dumps(payload, indent=2, default=numeric, allow_nan=False) + "\n"
        )

    def _milestone_id(self, stage: str) -> str:
        return self.stages[stage].milestone_id

    def _wait(self, reason: str) -> ToolCall:
        self.wait_steps += 1
        if self.wait_steps > 8:
            raise RuntimeError(
                f"Reference stalled awaiting measured evidence: {reason}"
            )
        return ToolCall(tool="wait", args={"seconds": 0.5})

    def _advance(self, phase: str) -> None:
        self.phase = phase
        self.wait_steps = 0
        self.best_heading_error_deg = math.inf
        self.heading_stall_decisions = 0
        self.navigation_recovery_attempts = 0

    def _navigate(
        self,
        observation: Any,
        goal: tuple[float, float],
        phase: str,
        *,
        arrival_radius_m: float = PHYSICAL_TASK_NAVIGATION_TOLERANCE_M,
    ) -> ToolCall | None:
        pose = observation.pose
        x, y = pose["position"][:2]
        dx, dy = goal[0] - x, goal[1] - y
        if not math.isfinite(arrival_radius_m) or arrival_radius_m <= 0:
            raise ValueError("Navigation arrival radius must be finite and positive.")
        distance_m = math.hypot(dx, dy)
        if distance_m <= arrival_radius_m:
            self._advance(phase)
            return None
        # A goal-region A* may snap to a neighbouring free cell whose path is
        # already inside the handler deadband while the measured body remains
        # just outside this stricter episode arrival radius.  Close only that
        # terminal one-cell gap against the reviewed endpoint.  The exact
        # helper still requires the unchanged inflated-grid A* and a hard-free
        # segment; longer legs retain ordinary goal-region planning.
        if distance_m <= arrival_radius_m + LOCOMOTION_MIN_STEP_DISTANCE_M:
            return self._heading_preserving_pointgoal(
                observation,
                goal,
                max_step_m=distance_m,
            )
        return self._pointgoal(observation, goal)

    def _pointgoal(self, observation: Any, goal: tuple[float, float]) -> ToolCall:
        pose = observation.pose
        x, y = pose["position"][:2]
        dx, dy = goal[0] - x, goal[1] - y
        yaw = pose["yaw_rad"]
        return ToolCall(
            tool="navigate_to_pointgoal",
            args={
                "forward_m": math.cos(yaw) * dx + math.sin(yaw) * dy,
                "left_m": -math.sin(yaw) * dx + math.cos(yaw) * dy,
                "target_description": f"{self.phase}: frozen reference endpoint",
            },
        )

    def _clearance_buffer_exit(
        self,
        observation: Any,
        context: str,
        *,
        preferred_goal: Any = None,
    ) -> ToolCall:
        """Use the shared measured-pose exit; leave the task endpoint unchanged."""
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if preferred_goal is None:
            endpoint = self.grid.clearance_buffer_exit(actual)
        else:
            endpoint = self.grid.clearance_buffer_exit(
                actual,
                preferred_goal_xy=np.asarray(preferred_goal, dtype=float),
            )
        if endpoint is None:
            raise RuntimeError(
                f"Measured {context} is blocked; no collision-monotone buffer exit."
            )
        return self._pointgoal(observation, endpoint)

    def _orient(
        self,
        observation: Any,
        xy: Any,
        phase: str,
        *,
        tolerance_deg: float = PHYSICAL_TASK_HEADING_TOLERANCE_DEG,
    ) -> ToolCall | None:
        pose = observation.pose
        x, y = pose["position"][:2]
        desired = math.atan2(xy[1] - y, xy[0] - x)
        error = math.degrees(
            (desired - pose["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
        )
        if abs(error) <= tolerance_deg:
            self._advance(phase)
            return None
        precheck_alignment = (
            self.fetch
            and self.phase in {"pregrasp_alignment", "grasp_alignment"}
            and self._measure_grasp_stance(observation).position_valid
        )
        if not self.grid.is_navigable(np.asarray([x, y])) and not precheck_alignment:
            return self._clearance_buffer_exit(observation, "orientation pose")
        # Oscillating against furniture is not progress toward the requested
        # heading, even when total angular travel is large. This uses the
        # shared no-progress threshold, not an episode/physics step budget.
        if self.best_heading_error_deg - abs(error) >= math.degrees(
            NAVIGATION_MIN_HEADING_PROGRESS_RAD
        ):
            self.best_heading_error_deg = abs(error)
            self.heading_stall_decisions = 0
        else:
            self.heading_stall_decisions += 1
            if self.heading_stall_decisions >= NAVIGATION_STUCK_TERMINATION_DECISIONS:
                raise RuntimeError(
                    f"Reference orientation stalled in {self.phase}: "
                    f"heading_error_deg={abs(error):.3f}"
                )
        maximum_turn_deg = min(
            LOCOMOTION_MAX_TURN_DEG,
            math.degrees(NAVIGATION_SEGMENT_MAX_TURN_RAD),
        )
        command_deg = float(np.clip(error, -maximum_turn_deg, maximum_turn_deg))
        if abs(command_deg) < LOCOMOTION_MIN_TURN_DEG:
            command_deg = math.copysign(LOCOMOTION_MIN_TURN_DEG, command_deg)
        return ToolCall(
            tool="move_body",
            args={
                "delta": [0.0, 0.0, command_deg],
            },
        )

    def _final_grasp_approach(self, observation: Any) -> ToolCall | None:
        """Use bounded shared PointGoals, preserving the pregrasp heading."""
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        stance = self._measure_grasp_stance(observation)
        if stance.valid:
            # A* continues to require the full 0.25 m planning footprint.  At
            # the manipulation boundary, however, a robot that is already
            # physically stable, raw-cell clear and inside the exact precheck
            # tolerance must not be pushed into an exit/re-entry oscillation
            # solely because gait settling touched the conservative buffer.
            self._advance("target_gaze")
            return None
        if stance.position_valid:
            # Reproduce the heading used by the direct physical precheck. A
            # small in-place correction here is preferable to launching
            # cuRobo from a merely authoring-valid orientation.
            self._advance("pregrasp_alignment")
            return self._orient(
                observation,
                self._live_target_xy(),
                "target_gaze",
                tolerance_deg=REFERENCE_GRASP_HEADING_TOLERANCE_DEG,
            )
        if not self.grid.is_navigable(actual):
            return self._clearance_buffer_exit(observation, "pregrasp body position")
        return self._center_grasp_stance(observation)

    def _center_grasp_stance(self, observation: Any) -> ToolCall:
        """Re-establish the measured stance by bounded XY, keeping camera gaze."""
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if not self.grid.is_navigable(actual):
            return self._clearance_buffer_exit(observation, "pregrasp body position")
        target_xy = self._live_target_xy()
        authored_stance = np.asarray(self.target_stance, dtype=float)
        authored_distance = float(np.linalg.norm(target_xy - authored_stance))
        # Direct grasp certification executes at the authored B pose.  Long
        # locomotion can stop several centimetres inside the public 0.12 m B
        # authoring region, which is spatially valid but not the pose that was
        # proven by SAM3/GraspGenX/cuRobo.  Recover that exact launch pose first.
        if (
            np.linalg.norm(actual - authored_stance)
            > REFERENCE_PRECHECK_STANCE_TOLERANCE_M
            and FETCH_STANCE_DISTANCE_BAND_M[0]
            <= authored_distance
            <= FETCH_STANCE_DISTANCE_BAND_M[1]
            and self.grid.is_navigable(authored_stance)
        ):
            return self._heading_preserving_pointgoal(observation, authored_stance)
        # B is a bounded stance region, not a required exact waypoint. Aim at
        # its reviewed stand-off distance on the measured body-heading ray.
        # Do not pull an intentionally outer stance back toward a counter by
        # replacing it with the candidate envelope's midpoint. Correct residual
        # bearing by short XY translation, not by a
        # table-side turn whose gait drift can destroy a valid reach stance.
        yaw = float(observation.pose["yaw_rad"])
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        distance = float(
            np.clip(
                np.linalg.norm(target_xy - self.target_stance),
                *FETCH_STANCE_DISTANCE_BAND_M,
            )
        )
        grasp_goal = target_xy - forward * distance
        # If gait drift leaves the body just outside the reach band, aiming at
        # an authored stance on the band edge can demand a sub-resolution
        # correction. Move the resolved goal one local-motion resolution into
        # the same admissible band so the correction is executable; do not
        # widen the band or move the original B region.
        correction_m = float(np.linalg.norm(grasp_goal - actual))
        actual_distance_m = float(np.linalg.norm(target_xy - actual))
        low, high = FETCH_STANCE_DISTANCE_BAND_M
        if correction_m < REFERENCE_GRASP_GOAL_RADIUS_M:
            if actual_distance_m > high:
                distance = min(distance, high - REFERENCE_GRASP_GOAL_RADIUS_M)
            elif actual_distance_m < low:
                distance = max(distance, low + REFERENCE_GRASP_GOAL_RADIUS_M)
            grasp_goal = target_xy - forward * distance
        if np.linalg.norm(
            grasp_goal - self.target_stance
        ) > REFERENCE_PRECHECK_STANCE_TOLERANCE_M or not self.grid.is_navigable(
            grasp_goal
        ):
            grasp_goal = self._feasible_grasp_heading_goal(target_xy, forward)
        return self._heading_preserving_pointgoal(observation, grasp_goal)

    def _alternate_grasp_stance(self, observation: Any) -> ToolCall | None:
        """Move within the authored B region after a pose-level grasp failure.

        A closing pose can establish contact yet prove unsuitable for the
        measured carry transition near a joint limit. Repeating SAM and
        GraspGenX from the identical body pose only regenerates the same
        kinematic envelope. Rotate the target-to-stance ray by a bounded
        fraction of the existing heading tolerance instead. The target,
        conservative reach band, authored B tolerance, occupancy footprint,
        and strict manipulation gates all remain unchanged.
        """

        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        target_xy = self._live_target_xy()
        authored = np.asarray(self.target_stance, dtype=float)
        radial = authored - target_xy
        distance = float(np.linalg.norm(radial))
        if (
            not FETCH_STANCE_DISTANCE_BAND_M[0]
            <= distance
            <= FETCH_STANCE_DISTANCE_BAND_M[1]
        ):
            raise RuntimeError(
                "Cannot recover grasp stance outside the conservative reach band."
            )
        # A failed cuRobo solve can report the smallest body-forward correction
        # that made the same public grasp candidates solvable. Execute a
        # resolvable version of that measured correction first, while keeping
        # the new stance inside the original authored B region and reach band.
        # This avoids spending all three physical attempts at an identical
        # known-infeasible pose.
        candidates: list[np.ndarray] = []
        right_hand = (
            getattr(
                self,
                "grasp_reachability_feedback",
                {},
            ).get("right_hand")
            or {}
        )
        reach_correction = right_hand.get("reach_correction") or {}
        if reach_correction.get("status") == "measured":
            measured = float(reach_correction["body_forward_m"])
            executable = max(measured, 2 * REFERENCE_GRASP_GOAL_RADIUS_M)
            executable = min(
                executable,
                PHYSICAL_TASK_NAVIGATION_TOLERANCE_M,
                max(0.0, distance - FETCH_STANCE_DISTANCE_BAND_M[0]),
            )
            if executable > REFERENCE_GRASP_GOAL_RADIUS_M:
                candidate = authored - radial * (executable / distance)
                if (
                    np.linalg.norm(candidate - authored)
                    <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
                    and self.grid.is_navigable(candidate)
                    and self.grid._planner().is_line_collision_free(actual, candidate)
                ):
                    candidates.append(candidate)
        # Clockwise first, then its mirror. The ordering is embodiment-wide
        # and deterministic; every endpoint must independently pass the same
        # B-region, footprint and swept-segment checks.
        fraction_order = (-0.5, 0.5, -0.9, 0.9)
        for fraction in fraction_order:
            angle = math.radians(PHYSICAL_TASK_HEADING_TOLERANCE_DEG * fraction)
            rotation = np.asarray(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            candidate = target_xy + rotation @ radial
            if (
                np.linalg.norm(candidate - authored)
                <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
                and self.grid.is_navigable(candidate)
                and self.grid._planner().is_line_collision_free(actual, candidate)
            ):
                candidates.append(candidate)
        if not candidates:
            # Some authored B regions contain exactly one collision-free body
            # pose (for example, a narrow checkout counter).  A pose-level
            # grasp failure still excludes the failed closing centre inside
            # the shared grasp handler, so a fresh SAM3/GraspGenX/cuRobo pass
            # from the same valid stance is a distinct physical retry.  Do
            # not turn the absence of a second stance into an early stop that
            # bypasses the configured three-execution limit.
            stance = self._measure_grasp_stance(observation)
            if not stance.valid:
                raise RuntimeError(
                    "No collision-free alternate grasp stance exists inside "
                    "the original B region and the current stance is no "
                    "longer valid."
                )
            self._advance("grasp_alignment")
            return None
        index = min(self.grasp_stance_recovery_attempts - 1, len(candidates) - 1)
        goal = candidates[index]
        self.active_grasp_stance = tuple(float(value) for value in goal)
        if np.linalg.norm(actual - goal) <= REFERENCE_GRASP_GOAL_RADIUS_M:
            self._advance("grasp_alignment")
            return None
        return self._heading_preserving_pointgoal(observation, goal)

    def _feasible_grasp_heading_goal(
        self, target_xy: np.ndarray, forward: np.ndarray
    ) -> np.ndarray:
        """Search legal reach intervals within the existing heading tolerance.

        Keep B, reach distance and occupied cells unchanged. Prefer the central
        ray, but do not reject a legal near-forward stance after small gait yaw
        drift merely because its bearing is not mathematically zero.
        """
        offset = target_xy - np.asarray(self.target_stance, dtype=float)
        mask = self.grid.planning_navigable_mask()
        for fraction in (0.0, -0.5, 0.5, -0.9, 0.9):
            angle = math.radians(PHYSICAL_TASK_HEADING_TOLERANCE_DEG * fraction)
            direction = np.array(
                [
                    math.cos(angle) * forward[0] - math.sin(angle) * forward[1],
                    math.sin(angle) * forward[0] + math.cos(angle) * forward[1],
                ]
            )
            projection = float(np.dot(offset, direction))
            perpendicular_sq = max(0.0, float(np.dot(offset, offset)) - projection**2)
            residual = PHYSICAL_TASK_NAVIGATION_TOLERANCE_M**2 - perpendicular_sq
            if residual <= 0:
                continue
            half_span = math.sqrt(residual)
            low = max(FETCH_STANCE_DISTANCE_BAND_M[0], projection - half_span)
            high = min(FETCH_STANCE_DISTANCE_BAND_M[1], projection + half_span)
            if high <= low:
                continue
            near, far = target_xy - direction * low, target_xy - direction * high
            cuts, cells = world_segment_cell_intervals(self.grid, near, far, mask.shape)
            intervals: list[list[float]] = []
            for left, right, (row, col) in zip(cuts[:-1], cuts[1:], cells):
                if not (
                    0 <= row < mask.shape[0]
                    and 0 <= col < mask.shape[1]
                    and mask[row, col]
                ):
                    continue
                if intervals and abs(intervals[-1][1] - left) <= 1e-12:
                    intervals[-1][1] = float(right)
                else:
                    intervals.append([float(left), float(right)])
            for left, right in sorted(
                intervals, key=lambda pair: pair[1] - pair[0], reverse=True
            ):
                goal = near + (far - near) * ((left + right) / 2)
                if (
                    self.grid.is_navigable(goal)
                    and np.linalg.norm(goal - self.target_stance)
                    <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
                ):
                    return goal
        raise RuntimeError(
            "No body-clear conservative reach interval inside the original B region and heading tolerance."
        )

    def _heading_preserving_pointgoal(
        self,
        observation: Any,
        goal_xy: Any,
        *,
        max_step_m: float = REFERENCE_FINAL_APPROACH_STEP_M,
        preserve_heading: bool | None = None,
    ) -> ToolCall:
        """Execute a bounded A* prefix; manipulation explicitly preserves yaw.

        Navigation-only approaches may span several corners and must allow
        the shared handler to align in a clear place. Exact endpoints alone
        do not imply a heading lock.
        """
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if not self.grid.is_navigable(actual):
            return self._clearance_buffer_exit(
                observation,
                "table-side body position",
                preferred_goal=goal_xy,
            )
        handler = self.tools.handler("navigate_to_pointgoal")
        if handler is not None:
            handler.configure_goal_region(
                search_radius_m=REFERENCE_GRASP_GOAL_RADIUS_M,
                exact_endpoint=True,
                preserve_heading=(
                    self.fetch if preserve_heading is None else preserve_heading
                ),
            )
        # This endpoint was selected from the admissible stance region already.
        # A second region snap can shorten a legal correction below the shared
        # execution resolution. Match the public handler's exact-endpoint mode.
        path, reason = self.grid.astar_with_reason(actual, goal_xy)
        if path is None or len(path) < 2:
            raise RuntimeError(f"No table-side A* segment: {reason}")
        planner = self.grid._planner()
        requested = np.asarray(goal_xy, dtype=float)
        # A* remains the reachability authority.  For a short table-side leg,
        # however, a hard-clear chord is safer for the physical gait than a
        # soft-clearance detour made of short lateral/backward corners.  The
        # unchanged 0.25 m inflated mask still gates this shortcut.
        if planner.is_line_collision_free(actual, requested):
            waypoint = requested
        else:
            waypoint = np.asarray(path[1])
            if np.linalg.norm(waypoint - actual) < REFERENCE_GRASP_GOAL_RADIUS_M:
                waypoint = planner.executable_prefix(
                    path,
                    REFERENCE_GRASP_GOAL_RADIUS_M,
                )
        segment = waypoint - actual
        length = float(np.linalg.norm(segment))
        if not math.isfinite(length) or length <= 0:
            raise RuntimeError("Table-side A* segment is not a finite displacement.")
        if not math.isfinite(max_step_m) or max_step_m <= 0:
            raise ValueError("Table-side maximum step must be finite and positive.")
        goal = actual + segment * min(1.0, max_step_m / length)
        if not self.grid._planner().is_line_collision_free(actual, goal):
            raise RuntimeError("Measured table-side segment crosses blocked geometry.")
        return self._pointgoal(observation, tuple(goal))

    def _grasp_departure(self, observation: Any) -> ToolCall | None:
        """Leave the worktop on A* before allowing a carrying return turn."""
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if not self.grid.is_navigable(actual):
            return self._clearance_buffer_exit(observation, "grasp departure")
        if (
            np.linalg.norm(actual - self.departure_stance)
            <= REFERENCE_LOADED_DEPARTURE_GOAL_RADIUS_M
        ):
            handler = self.tools.handler("navigate_to_pointgoal")
            if handler is not None:
                handler.configure_goal_region(
                    search_radius_m=REFERENCE_LOADED_DEPARTURE_GOAL_RADIUS_M,
                )
            self._advance("return_or_delivery")
            return None
        return self._heading_preserving_pointgoal(
            observation,
            self.departure_stance,
            max_step_m=REFERENCE_LOADED_DEPARTURE_STEP_M,
        )

    def _task_departure(self, observation: Any) -> ToolCall | None:
        """Leave an inspection site along A* before any return turn."""
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if not self.grid.is_navigable(actual):
            return self._clearance_buffer_exit(observation, "task departure")
        if (
            np.linalg.norm(actual - self.task_departure_stance)
            <= REFERENCE_NAVIGATION_GOAL_RADIUS_M
        ):
            handler = self.tools.handler("navigate_to_pointgoal")
            if handler is not None:
                handler.configure_goal_region(
                    search_radius_m=REFERENCE_NAVIGATION_GOAL_RADIUS_M,
                )
            self._advance("return_or_delivery")
            return None
        return self._heading_preserving_pointgoal(
            observation,
            self.task_departure_stance,
            preserve_heading=True,
        )

    def _recipient_can_receive(self) -> bool:
        """Use the live shared simulator predicate, never a sticky milestone."""
        state = (self.world_snapshot.get("humans") or {}).get(self.recipient_id)
        visible_contact = bool(
            isinstance(state, dict)
            and state.get("robot_can_communicate") is True
            and state.get("human_visible_to_robot") is True
        )
        if not visible_contact or not self.fetch:
            return visible_contact
        # Speaking range is not handover reach. Use full-precision live poses,
        # not the snapshot's two-decimal display distance near the 1 m gate.
        robot_position = (self.world_snapshot.get("robot") or {}).get("position")
        human_position = state.get("position")
        if robot_position is None or human_position is None:
            return False
        distance = horizontal_distance_m(robot_position, human_position)
        return HANDOVER_MIN_DISTANCE_M <= distance <= HANDOVER_MAX_DISTANCE_M

    def _recover_recipient_contact(self, observation: Any) -> ToolCall:
        """Re-center and re-orient after measured turn drift breaks contact."""
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if not self.grid.is_navigable(actual):
            return self._clearance_buffer_exit(observation, "recipient-contact pose")
        correction_tolerance = (
            REFERENCE_GRASP_GOAL_RADIUS_M
            if self.fetch
            else REFERENCE_NAVIGATION_GOAL_RADIUS_M
        )
        if np.linalg.norm(actual - self.completion_stance) > correction_tolerance:
            return self._heading_preserving_pointgoal(
                observation, self.completion_stance
            )
        call = self._orient(
            observation,
            self.recipient.start_position,
            "recipient_evidence",
        )
        if call is not None:
            return call
        raise RuntimeError(
            "Reviewed completion stance cannot establish live visible "
            "speaking contact with the recipient."
        )

    def _recover_repeated_navigation_contact(self, observation: Any) -> ToolCall | None:
        """Back out from repeated static contact before resuming the same A* leg.

        One reported touch is ordinary execution evidence. Two consecutive
        contacted PointGoal steps mean that continuing the same body sweep can
        accumulate a real fall even while the 0.25 m centre route remains
        valid. Choose a short raw-map A* endpoint opposite the measured contact
        bearing, preserve body heading, then replan the unchanged task goal.
        """
        if self.navigation_static_contact_streak < 2 or self.phase not in {
            "outbound",
            "return_or_delivery",
        }:
            return None
        # Arrival owns the phase boundary.  A contact reported by the final
        # PointGoal step is evidence about that completed motion, not a reason
        # to push the body back out of an already satisfied goal region before
        # `_navigate` can advance the state machine.  This applies equally to
        # the outbound hand-off and the return/delivery hand-off.
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if self.phase == "outbound":
            active_goal = (
                self.approach_stance if self.fetch else self.task_approach_stance
            )
            arrival_radius_m = REFERENCE_INTERMEDIATE_CHECKPOINT_RADIUS_M
        else:
            active_goal = self.completion_approach_stance
            arrival_radius_m = PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
        if (
            np.linalg.norm(actual - np.asarray(active_goal, dtype=float))
            <= arrival_radius_m
        ):
            return None
        contacts = [
            item
            for item in self.feedback.get("physical_contact_details", ())
            if item.get("source") == "static_scene"
            and math.isfinite(float(item.get("relative_bearing_deg", math.nan)))
        ]
        if not contacts:
            return None
        if not self.grid.is_navigable(actual):
            # Contact can push the measured centre just inside the inflated
            # planning buffer.  Reissuing the original PointGoal from that
            # pose bypasses the recovery promised here and repeats the same
            # blocked terminal segment indefinitely.  Route the already-
            # measured incident through the shared raw-free, collision-
            # monotone buffer exit before replanning the unchanged task goal.
            return self._clearance_buffer_exit(
                observation,
                "contact-displaced body position",
                preferred_goal=active_goal,
            )
        contact = min(
            contacts, key=lambda item: float(item.get("distance_m", math.inf))
        )
        away_yaw = (
            float(observation.pose["yaw_rad"])
            + math.radians(float(contact["relative_bearing_deg"]))
            + math.pi
        )
        candidates = [
            (offset_deg, distance_m)
            for offset_deg in (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0)
            for distance_m in (0.15, 0.10)
        ]
        # A scene contact can block the geometrically legal first backout even
        # though another A*-legal direction is physically executable. Cycle
        # through the same bounded candidate set after measured non-progress
        # instead of issuing one failed relative-pose command indefinitely.
        # Material measured progress resets this index in ``update_info``.
        start_index = self.navigation_recovery_attempts % len(candidates)
        for relative_index in range(len(candidates)):
            index = (start_index + relative_index) % len(candidates)
            offset_deg, distance_m = candidates[index]
            angle = away_yaw + math.radians(offset_deg)
            goal = actual + distance_m * np.array([math.cos(angle), math.sin(angle)])
            path, _ = self.grid.astar_with_reason(actual, goal)
            if path is not None and len(path) >= 2:
                self.navigation_recovery_attempts = index + 1
                # Keep the incident active until execution demonstrates a
                # material displacement. Clearing it when a backout is merely
                # commanded makes the next selection immediately re-enter the
                # original A* sweep.
                return self._heading_preserving_pointgoal(observation, goal)
        return None

    def _complete_physical_grasp(self, observation: Any) -> ToolCall | None:
        """Permit only local departure with live friction hold; certify after clearance.

        Retraction can leave a retained payload over the support edge. The
        existing A* departure is part of acquisition, not proof of transport
        readiness. Never admit a missing/dropped or constrained target.
        """
        state = self.world_snapshot["manipulation"]["objects"][self.target["object_id"]]
        hold = state.get("physical_hold", {})
        if (
            hold.get("held") is not True
            or hold.get("contact_established") is not True
            or hold.get("constraint_joint_present") is not False
        ):
            raise RuntimeError(
                f"Grasp acquisition has no live unconstrained friction hold: {hold}"
            )
        if hold.get("transport_ready") is True:
            return None
        phase = self.phase
        call = self._grasp_departure(observation)
        self.phase = phase
        if call is None:
            raise RuntimeError(
                f"Payload remains on its support after local departure: {hold}"
            )
        return call

    def _live_target_xy(self) -> np.ndarray:
        live = self.world_snapshot["manipulation"]["objects"][self.target["object_id"]]
        target_xy = (
            np.asarray(live["bbox_min"])[:2] + np.asarray(live["bbox_max"])[:2]
        ) / 2
        if not np.isfinite(target_xy).all():
            raise RuntimeError("Measured grasp stance target has non-finite bounds.")
        return target_xy

    def _measure_grasp_stance(self, observation: Any) -> MeasuredGraspStance:
        actual = np.asarray(observation.pose["position"][:2], dtype=float)
        if not np.isfinite(actual).all() or not math.isfinite(
            observation.pose["yaw_rad"]
        ):
            raise RuntimeError("Measured grasp stance contains a non-finite pose.")
        target_xy = self._live_target_xy()
        distance = float(np.linalg.norm(actual - target_xy))
        expected_stance = np.asarray(
            self.active_grasp_stance
            if self.active_grasp_stance is not None
            else self.target_stance,
            dtype=float,
        )
        stance_error = float(np.linalg.norm(actual - expected_stance))
        desired = math.atan2(target_xy[1] - actual[1], target_xy[0] - actual[0])
        heading_error = abs(
            math.degrees(
                (desired - observation.pose["yaw_rad"] + math.pi) % (2 * math.pi)
                - math.pi,
            )
        )
        within_precheck_pose = (
            stance_error <= REFERENCE_PRECHECK_STANCE_TOLERANCE_M
            and np.linalg.norm(actual - self.target_stance)
            <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
        )
        raw_free = getattr(self.grid, "is_raw_free", lambda _xy: False)(actual)
        body_clear = self.grid.is_navigable(actual) or (
            within_precheck_pose and raw_free
        )
        return MeasuredGraspStance(body_clear, stance_error, distance, heading_error)

    def _require_grasp_stance(self, observation: Any) -> None:
        stance = self._measure_grasp_stance(observation)
        if not stance.valid:
            raise RuntimeError(
                f"Measured grasp stance invalid: {stance}; no SAM or physical grasp executed."
            )

    def _select(self, observation: Any) -> tuple[ToolCall, dict[str, Any]]:
        contact_recovery = self._recover_repeated_navigation_contact(observation)
        if contact_recovery is not None:
            return contact_recovery, {
                "navigation_recovery": "repeated_static_contact_backout"
            }
        for _ in range(16):
            if self.phase == "dispatch":
                if self._milestone_id("plan") not in self.opened:
                    return self._wait("initial request"), {}
                self._advance("outbound")
            elif self.phase == "outbound":
                goal = self.approach_stance if self.fetch else self.task_approach_stance
                next_phase = "pregrasp_heading" if self.fetch else "pretask_heading"
                call = self._navigate(
                    observation,
                    goal,
                    next_phase,
                    arrival_radius_m=REFERENCE_INTERMEDIATE_CHECKPOINT_RADIUS_M,
                )
                if call:
                    return call, {}
            elif self.phase == "pretask_heading":
                x, y = observation.pose["position"][:2]
                call = self._orient(
                    observation,
                    (x + math.cos(self.task_yaw), y + math.sin(self.task_yaw)),
                    "final_task_approach",
                )
                if call:
                    return call, {}
            elif self.phase == "final_task_approach":
                actual = np.asarray(observation.pose["position"][:2], dtype=float)
                if (
                    np.linalg.norm(actual - self.target_stance)
                    <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
                ):
                    self._advance("target_heading")
                    continue
                return self._heading_preserving_pointgoal(
                    observation, self.target_stance
                ), {}
            elif self.phase == "pregrasp_heading":
                x, y = observation.pose["position"][:2]
                call = self._orient(
                    observation,
                    (x + math.cos(self.grasp_yaw), y + math.sin(self.grasp_yaw)),
                    "final_grasp_approach",
                )
                if call:
                    return call, {}
            elif self.phase == "pregrasp_alignment":
                call = self._orient(
                    observation,
                    self._live_target_xy(),
                    "target_gaze",
                    tolerance_deg=REFERENCE_GRASP_HEADING_TOLERANCE_DEG,
                )
                if call:
                    return call, {}
            elif self.phase == "final_grasp_approach":
                call = self._final_grasp_approach(observation)
                if call:
                    return call, {}
            elif self.phase == "target_heading":
                target_position = (
                    self.source.start_position
                    if self.inquiry
                    else self.target["position_xyz"]
                )
                call = self._orient(observation, target_position, "target_gaze")
                if call:
                    return call, {}
            elif self.phase == "target_gaze":
                self._advance("target_evidence")
                # Close-range grasping needs a downward view. Applying that
                # offset to navigation inspections can crop elevated targets.
                pitch = GRASP_CERTIFICATION_CAMERA_TILT_DEG if self.fetch else 0.0
                return ToolCall(tool="tilt_camera", args={"pitch_deg": pitch}), {}
            elif self.phase == "target_evidence":
                if self._milestone_id("discovery") not in self.completed:
                    return self._wait("target visible in the current ego RGB-D"), {}
                self._advance(
                    "grasp"
                    if self.fetch
                    else "inquiry"
                    if self.inquiry
                    else "inspection"
                )
            elif self.phase == "inquiry":
                self._advance("inquiry_evidence")
                return ToolCall(
                    tool="ask",
                    args={
                        "recipient": self.source.name,
                        "intent_id": self.inquiry_intent.intent_id,
                        "message": self.inquiry_intent.description,
                    },
                ), {}
            elif self.phase == "inquiry_evidence":
                if self._milestone_id("onsite") not in self.completed:
                    return self._wait(
                        "actual answer received from the onsite source"
                    ), {}
                self._advance("restore_gaze")
            elif self.phase == "inspection":
                if self._milestone_id("onsite") not in self.opened:
                    return self._wait("onsite inspection event"), {}
                answer = self.stages["onsite"].outcome.ground_truth["answer"]
                self._advance("inspection_evidence")
                return ToolCall(tool="wait", args={"seconds": 0.5}), {
                    "inspection": answer
                }
            elif self.phase == "inspection_evidence":
                if self._milestone_id("onsite") not in self.completed:
                    return self._wait("accepted inspection submission"), {}
                self._advance("restore_gaze")
            elif self.phase == "grasp":
                if self.grasp_executions >= REFERENCE_GRASP_EXECUTION_LIMIT:
                    raise RuntimeError(
                        "Reference exhausted its bounded physical grasp executions."
                    )
                # The gaze action advances real physics. If those ticks move
                # a previously legal stance outside its band, use the same XY
                # approach before the first SAM request, without tilting again.
                stance = self._measure_grasp_stance(observation)
                if not stance.position_valid:
                    return self._center_grasp_stance(observation), {}
                if not stance.valid:
                    self._advance("grasp_alignment")
                    call = self._orient(
                        observation,
                        self._live_target_xy(),
                        "grasp",
                        tolerance_deg=REFERENCE_GRASP_HEADING_TOLERANCE_DEG,
                    )
                    if call is not None:
                        return call, {}
                self._require_grasp_stance(observation)
                self._advance("hold_evidence")
                return ToolCall(
                    tool="grasp_object", args={"target": self.grasp_query()}
                ), {}
            elif self.phase == "grasp_alignment":
                call = self._orient(
                    observation,
                    self._live_target_xy(),
                    "grasp",
                    tolerance_deg=REFERENCE_GRASP_HEADING_TOLERANCE_DEG,
                )
                if call:
                    return call, {}
            elif self.phase == "grasp_retry_stance":
                call = self._alternate_grasp_stance(observation)
                if call:
                    return call, {}
            elif self.phase == "hold_evidence":
                call = self._complete_physical_grasp(observation)
                if call:
                    return call, {}
                self._advance("restore_gaze")
            elif self.phase == "restore_gaze":
                handler = self.tools.handler("navigate_to_pointgoal")
                if handler is not None:
                    handler.configure_goal_region(
                        search_radius_m=REFERENCE_NAVIGATION_GOAL_RADIUS_M
                    )
                self._advance("grasp_departure" if self.fetch else "task_departure")
                return ToolCall(tool="tilt_camera", args={"pitch_deg": 0.0}), {}
            elif self.phase == "grasp_departure":
                call = self._grasp_departure(observation)
                if call:
                    return call, {}
            elif self.phase == "task_departure":
                call = self._task_departure(observation)
                if call:
                    return call, {}
            elif self.phase == "return_or_delivery":
                call = self._navigate(
                    observation, self.completion_approach_stance, "prerecipient_heading"
                )
                if call:
                    return call, {}
            elif self.phase == "prerecipient_heading":
                x, y = observation.pose["position"][:2]
                call = self._orient(
                    observation,
                    (
                        x + math.cos(self.completion_yaw),
                        y + math.sin(self.completion_yaw),
                    ),
                    "final_recipient_approach",
                )
                if call:
                    return call, {}
            elif self.phase == "final_recipient_approach":
                actual = np.asarray(observation.pose["position"][:2], dtype=float)
                if (
                    np.linalg.norm(actual - self.completion_stance)
                    <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
                ):
                    handler = self.tools.handler("navigate_to_pointgoal")
                    if handler is not None:
                        handler.configure_goal_region(
                            search_radius_m=REFERENCE_NAVIGATION_GOAL_RADIUS_M
                        )
                    self._advance("recipient_heading")
                    continue
                return self._heading_preserving_pointgoal(
                    observation, self.completion_stance
                ), {}
            elif self.phase == "recipient_heading":
                if self._recipient_can_receive():
                    self._advance("recipient_evidence")
                    continue
                call = self._orient(
                    observation, self.recipient.start_position, "recipient_evidence"
                )
                if call:
                    return call, {}
            elif self.phase == "recipient_evidence":
                if (
                    self._milestone_id("recipient") not in self.completed
                    or self._milestone_id("completion") not in self.opened
                ):
                    return self._wait(
                        "visible recipient and opened completion event"
                    ), {}
                # The interaction implementation is the authority for the
                # live visibility, distance, hold, and support-clearance
                # verdict.  Do not let a decision-boundary snapshot veto that
                # measured claim after the Robot has already reached the
                # reviewed completion stance.  In particular, an authored
                # stance can lie inside the occupancy grid's conservative
                # inflated buffer; repeatedly leaving that buffer and
                # returning to the same endpoint never creates new evidence.
                # A rejected claim is still recorded by the ordinary runtime
                # evaluator and consumes one of the three attempts below.
                actual = np.asarray(observation.pose["position"][:2], dtype=float)
                at_reviewed_stance = bool(
                    np.linalg.norm(actual - self.completion_stance)
                    <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M
                )
                if not self._recipient_can_receive() and not at_reviewed_stance:
                    return self._recover_recipient_contact(observation), {}
                self.completion_attempts += 1
                if self.completion_attempts > 3:
                    raise RuntimeError(
                        "Reference could not deliver to the live recipient after three measured attempts."
                    )
                self._advance("completion_evidence")
                if self.fetch:
                    return ToolCall(
                        tool="handover_object", args={"recipient": self.recipient.name}
                    ), {}
                return ToolCall(
                    tool="speak",
                    args={
                        "recipient": self.recipient.name,
                        "message": self.stages["completion"].outcome.ground_truth[
                            "answer"
                        ],
                    },
                ), {}
            elif self.phase == "completion_evidence":
                if self._milestone_id("completion") not in self.completed:
                    self._advance("recipient_evidence")
                    continue
                return self._wait("runner terminal success after report/handover"), {}
            else:
                raise ValueError(f"Unknown reference phase: {self.phase}")
        raise RuntimeError("Reference phase transition cycle.")

    def _record_observation(self, raw: dict[str, Any]) -> Any:
        observation = agent_observation_from_runtime(
            raw, camera_views=self.agent_harness_config.protocol.camera_views
        )
        ego = observation.views["ego"]
        if ego.depth is None or "capture_id" not in ego.camera_params:
            raise RuntimeError("Reference requires stamped, synchronized ego RGB-D.")
        stem = f"{self.step:04d}_{self.phase}"
        rgb = np.asarray(ego.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                "Reference needs native uint8 RGB, not a normalized or resized image."
            )
        if np.asarray(ego.depth).shape != rgb.shape[:2]:
            raise ValueError("Reference RGB and depth dimensions disagree.")
        Image.fromarray(rgb).save(self.output / f"{stem}.png")
        np.save(self.output / f"{stem}.depth.npy", np.asarray(ego.depth))
        self._save(
            f"{stem}.camera.json",
            {
                "camera_params": ego.camera_params,
                "pose": observation.pose,
                "native_resolution_wh": [rgb.shape[1], rgb.shape[0]],
                "opened": sorted(self.opened),
                "completed": sorted(self.completed),
            },
        )
        return observation

    def _request_grasp_preview(
        self,
        call: ToolCall,
        context: dict[str, Any],
        *,
        archive_tag: str = "",
        allow_not_in_view: bool = False,
    ) -> Any:
        """Execute the one-shot grasp and archive its public evidence."""
        if call.tool != "grasp_object":
            raise ValueError("Grasp collection must call grasp_object.")
        result = self.tools.execute(call, context)
        self._record_tool(call, result, f"request{archive_tag}")
        if result.is_error and not (
            allow_not_in_view and result.metadata.get("error_type") == "not_in_view"
        ):
            raise RuntimeError(f"Reference tool {call.tool} failed: {result.content}")
        if not result.is_error and not result.images:
            raise RuntimeError("Grasp result contains no current-frame overlay.")
        return result

    def _archive_grasp_overlays(
        self,
        result: Any,
        *,
        archive_tag: str = "",
    ) -> None:
        """Archive the current-frame overlays supplied by the grasp tool."""
        for index, data_url in enumerate(result.images):
            header, data = data_url.split(",", 1)
            extensions = {
                "data:image/png;base64": "png",
                "data:image/jpeg;base64": "jpg",
            }
            if header not in extensions:
                raise ValueError(f"Unsupported grasp overlay encoding: {header}")
            # Keep the bytes supplied by the shared tool, including encoding.
            extension = extensions[header]
            (
                self.output
                / (f"{self.step:04d}_grasp_overlay{archive_tag}_{index}.{extension}")
            ).write_bytes(base64.b64decode(data, validate=True))

    def _grasp_matches_gt_target(
        self,
        result: Any,
        *,
        archive_tag: str = "",
    ) -> bool:
        """Judge the executed grasp against the live GT target bounds.

        GT is used only for this judgment. It never replaces SAM3 pixels or
        supplies a pose to GraspGenX.
        """

        live_target = self.world_snapshot["manipulation"]["objects"][
            self.target["object_id"]
        ]
        lo = np.asarray(live_target["bbox_min"])
        hi = np.asarray(live_target["bbox_max"])
        center = np.asarray(result.metadata.get("target_center_world_m") or [()])
        matched = bool(
            center.shape == (3,)
            and np.isfinite(center).all()
            and np.all((center >= lo) & (center <= hi))
        )
        self._save(
            f"{self.step:04d}_grasp_gt_check{archive_tag}.json",
            {
                "source": "GT judgment over the executed SAM3 candidate",
                "measured_center_world": (
                    center.tolist() if center.shape == (3,) else None
                ),
                "matches_gt_target": matched,
            },
        )
        return matched

    def _execute(self, call: ToolCall, context: dict[str, Any]) -> list[Any]:
        if call.tool == "grasp_object":
            result = self._request_grasp_preview(
                call,
                context,
                allow_not_in_view=True,
            )
            results = [result]
            if result.is_error:
                # SAM may understand the public category but miss a composed
                # attribute phrase (for example, "blue-label bottle"). A
                # category fallback is allowed only when that same category is
                # literally present in the public referent. It repeats the
                # shared SAM3 tool on the same observation; it does not supply
                # a GT mask, execute a grasp, or consume a physical retry.
                category = str(self.target.get("category") or "").strip()
                public_name = self.specific_grasp_query()
                import re

                category_is_public = bool(
                    category
                    and category != call.args["target"]
                    and re.search(rf"\b{re.escape(category)}\b", public_name, re.I)
                )
                if not category_is_public:
                    raise RuntimeError(
                        f"Reference tool {call.tool} failed: {result.content}"
                    )
                category_call = ToolCall(
                    tool="grasp_object",
                    args={**call.args, "target": category},
                )
                result = self._request_grasp_preview(
                    category_call,
                    context,
                    archive_tag="_category",
                )
                # The first miss is fully archived above, but it must not be
                # serialized as an unhandled error beside the recovered
                # action. The episode runner correctly refuses any decision
                # containing an error ToolResult; only the successful public
                # category call owns this physical action step.
                results = [result]
                call = category_call
                self._archive_grasp_overlays(result, archive_tag="_category")
            else:
                self._archive_grasp_overlays(result)
        else:
            result = self.tools.execute(call, context)
            self._record_tool(call, result, "request")
            if result.is_error:
                raise RuntimeError(
                    f"Reference tool {call.tool} failed: {result.content}"
                )
            results = [result]
        if call.tool == "grasp_object":
            matched = self._grasp_matches_gt_target(result)
            specific_query = self.specific_grasp_query()
            if not matched and specific_query != call.args["target"]:
                # The public tool permits a more specific query after a
                # mis-grounded attempt. One refined retry, then the GT check
                # must pass before the rollout continues.
                refined = ToolCall(
                    tool="grasp_object",
                    args={**call.args, "target": specific_query},
                )
                result = self._request_grasp_preview(
                    refined,
                    context,
                    archive_tag="_refined",
                )
                self._archive_grasp_overlays(result, archive_tag="_refined")
                results.append(result)
                matched = self._grasp_matches_gt_target(
                    result,
                    archive_tag="_refined",
                )
                call = refined
            if not matched:
                raise RuntimeError(
                    "Executed SAM3 grasp candidate does not match the GT "
                    "target bounds.",
                )
            if result.is_error:
                raise RuntimeError(f"Reference grasp failed: {result.content}")
            self.grasp_executions += 1
        if result.action is None:
            raise RuntimeError(
                f"Reference tool produced no executable action: {call.tool}"
            )
        if (
            self.phase
            in {
                "final_grasp_approach",
                "final_task_approach",
                "grasp",
                "hold_evidence",
                "grasp_departure",
                "task_departure",
                "final_recipient_approach",
            }
            and self.fetch
        ):
            require_table_side_translation(result)
        return results

    def _record_tool(self, call: ToolCall, result: Any, stage: str) -> None:
        self._save(
            f"{self.step:04d}_tool_{stage}.json",
            {
                "call": {"name": call.tool, "args": call.args},
                "content": result.content,
                "is_error": result.is_error,
                "metadata": result.metadata,
            },
        )

    def finalize(self, runner: Any, result: dict[str, Any]) -> None:
        """Persist the same rollout, including diagnostic provenance on failure.

        GT trajectories remain ineligible for official policy scores. Closure
        is evidence for reference review, never an alias for scored success.
        """
        if not self._attempt_started or self.episode.episode_id != result["episode_id"]:
            return  # Bootstrap failure: the worker result owns the error.
        if result.get("decision_steps", 0):
            trajectory = runner.evaluation_trajectory()
            self._save("evaluation_trajectory.json", trajectory.model_dump(mode="json"))
            result["evaluation_trajectory_path"] = str(
                self.output / "evaluation_trajectory.json"
            )
        closed = bool(result.get("causal_graph_completed")) and "failure" not in result
        if self.fetch and not (
            1 <= self.grasp_executions <= REFERENCE_GRASP_EXECUTION_LIMIT
        ):
            closed = False
        result["reference_status"] = "closed_pending_review" if closed else "failed"
        self._save(
            "outcome.json",
            {
                "episode_id": self.episode.episode_id,
                "reference_status": result["reference_status"],
                "termination_reason": result["termination_reason"],
                "grasp_executions": self.grasp_executions,
                "physics_steps": result["physics_steps"],
                "semantic_answers_source": "authored_ground_truth",
                "official_policy_score_eligible": False,
                "failure": result.get("failure"),
            },
        )

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        action_space: list[str] | None = None,
        *,
        decision_requirements: Any = None,
        decision_budget: int | None = None,
    ) -> dict[str, Any]:
        del instruction, decision_budget
        current = self._record_observation(observation)
        call, fields = self._select(current)
        if action_space is not None and call.tool not in action_space:
            raise RuntimeError(f"Reference tool unavailable: {call.tool}")
        context = {
            "observation": current,
            "action_step": self.step,
            "online_occupancy_grid": self.grid,
            "record_dir": str(self.output),
            "feedback": self.feedback,
        }
        # Base Human placements are provisional authoring context for later
        # variants.  Keep their measured contacts in evidence, but do not make
        # GT A* diverge from the frozen base route to avoid those temporary
        # bodies.  Variant certification supplies its own finalized cast.
        results = self._execute(call, context)
        executed_args = dict(call.args)
        if call.tool == "grasp_object":
            # A same-frame public category/refinement retry can replace the
            # initially requested phrase. Keep the decision envelope aligned
            # with the action that will actually reach the environment.
            executed_target = results[-1].action.get("target")
            if executed_target:
                executed_args["target"] = executed_target
        envelope = {
            "thinking": f"GT reference phase: {self.phase}",
            "actions": [{"tool": call.tool, "args": executed_args}],
            **fields,
        }
        if (
            decision_requirements is not None
            and decision_requirements.planning is not None
        ):
            gt = self.stages["plan"].outcome.ground_truth
            envelope["plan"] = {"goal": gt["goal"], "steps": gt["required_subgoals"]}
        decision = {
            **results[-1].action,
            "envelope": envelope,
            "call_name": call.tool,
            "action_step": self.step,
            "policy_step": self.step,
            "policy_inference": True,
            "model_visible_action_steps": [self.step],
            "model_calls": [],
            "reference_policy": self.name,
            "reference_phase": self.phase,
            "diagnostic_only": True,
            "agent_harness_config": self.agent_harness_config.as_dict(),
            "tool_results": [
                {
                    "name": r.name,
                    "content": r.content,
                    "is_error": r.is_error,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }
        self._save(f"{self.step:04d}_decision.json", decision)
        self.last_call = call.tool
        self.step += 1
        return decision

    def update_info(self, info: dict[str, Any]) -> None:
        self.feedback = dict(info)
        navigation = info.get("navigation_execution") or {}
        progress = navigation.get("progress_m")
        if (
            self.phase == "grasp_departure"
            and self.last_call == "navigate_to_pointgoal"
            and str(navigation.get("target_description") or "").startswith(
                "grasp_departure:"
            )
            and progress is not None
            and math.isfinite(float(progress))
            and float(progress) <= 0.005
        ):
            self.loaded_departure_stall_decisions += 1
            if (
                self.loaded_departure_stall_decisions
                >= REFERENCE_LOADED_DEPARTURE_STALL_LIMIT
            ):
                raise RuntimeError(
                    "Loaded departure made no useful measured progress on two "
                    "closed-loop replans; refusing further carrying motion."
                )
        elif (
            self.phase == "grasp_departure"
            and self.last_call == "navigate_to_pointgoal"
        ):
            self.loaded_departure_stall_decisions = 0
        static_contact = bool(
            self.last_call == "navigate_to_pointgoal"
            and any(
                item.get("source") == "static_scene"
                for item in info.get("physical_contact_details", ())
            )
        )
        # Contact alone is not a navigation failure. If the measured body
        # displacement followed the requested A* segment, backing away from
        # that contact repeatedly undoes valid route progress (especially at
        # corners where Euclidean distance to the final goal can increase).
        motion = info.get("motion_outcome") or {}
        completion_ratio = motion.get("translation_completion_ratio")
        tracked_translation = (
            completion_ratio is not None
            and math.isfinite(float(completion_ratio))
            and float(completion_ratio) >= 0.5
        )
        material_progress = (
            self.navigation_recovery_attempts > 0
            and progress is not None
            and math.isfinite(float(progress))
            and float(progress) >= NAVIGATION_MIN_XY_PROGRESS_M
        )
        # An unloaded shoulder brush that still tracks the commanded segment
        # can be harmless route progress.  While carrying, however, repeated
        # static contact is a direct retention hazard even if the pelvis moves
        # the requested distance: the cabinet can load the torso/arm until the
        # next gait step ejects the object.  Preserve that incident so the
        # existing collision-checked backout runs before another loaded sweep.
        loaded_static_contact = static_contact and info.get("held") is True
        if (tracked_translation and not loaded_static_contact) or material_progress:
            self.navigation_static_contact_streak = 0
            self.navigation_recovery_attempts = 0
        elif static_contact:
            self.navigation_static_contact_streak += 1
        elif self.last_call != "navigate_to_pointgoal":
            self.navigation_static_contact_streak = 0
            self.navigation_recovery_attempts = 0
        else:
            # A short contact-free correction does not prove that the body has
            # left the same obstacle: the controller commonly alternates one
            # contacted step with a few-centimetre backout.  Preserve the
            # incident until A* makes material goal progress; this lets the
            # existing collision-checked backout handle intermittent contact
            # instead of orbiting the obstacle indefinitely.
            progress_m = float(navigation.get("progress_m") or 0.0)
            if math.isfinite(progress_m) and progress_m >= 0.15:
                self.navigation_static_contact_streak = 0
        if self.last_call == "grasp_object":
            self.physical_grasp_feedback = dict(info.get("grasp_skill", {}))
            self.grasp_reachability_feedback = dict(
                info.get("grasp_reachability") or {},
            )
        self._save(f"{self.step - 1:04d}_execution.json", info)
        if self.last_call == "grasp_object" and not info.get("executed"):
            failure = info.get("grasp_failure") or {}
            if self.grasp_executions < REFERENCE_GRASP_EXECUTION_LIMIT and failure.get(
                "code"
            ) in {
                "reach_envelope",
                "reach_alignment_failed",
                "contact_miss",
                "retention_lost",
            }:
                if failure.get("code") in {
                    "reach_envelope",
                    "reach_alignment_failed",
                    "retention_lost",
                }:
                    self.grasp_stance_recovery_attempts += 1
                    self._advance("grasp_retry_stance")
                else:
                    self._advance("grasp")
                return
            raise RuntimeError(f"Reference physical grasp failed: {info}")
