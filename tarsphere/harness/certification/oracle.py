"""Dataset-authoring oracle for Golden episode compilation and certification.

This agent is intentionally not an evaluated policy.  It may read episode GT
to choose semantic stops and the official occupancy layer to plan paths, but
all resulting actions still pass through the normal autonomous runner and
Isaac controllers.  It never teleports actors or mutates evaluator state.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from tarsphere.embodiment.dex3 import G1_MANIPULATION_HAND
from tarsphere.evaluation.schema import MilestoneCondition

from tarsphere import geometry as G
from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.bench.certification.grasp_reach import (
    GRASP_REACH_SEARCH,
    measure_grasp_reach,
)
from tarsphere.bench.certification.failures import (
    PhysicalTrialCandidateError,
)
from tarsphere.bench.certification.scene import (
    load_episode_occupancy,
)
from tarsphere.bench.spatial.physical_task.navigation_evidence import (
    g1_navigation_planner,
    reachable_radial_occupancy_points,
    require_exact_occupancy_point,
    route_reaches_exact_goal,
)
from tarsphere.infra.control.physical_contracts import (
    G1_ARM_IK_SOLVER_TOLERANCE_M,
    G1_CARRY_LOCOMOTION_UPPER_BODY_INTERMEDIATE_FRACTIONS,
    G1_CROUCH_GRASP_MAX_TARGET_TOP_M,
    G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_GRASP_ENTRY_MAX_FACING_ERROR_DEG,
    G1_GRASP_REACH_TOLERANCE_M,
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    G1_PRE_RELEASE_CLEARANCE_M,
    G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M,
    G1_ROTATION_POLICY_COMMAND,
    HANDOVER_MAX_DISTANCE_M,
    HANDOVER_MIN_DISTANCE_M,
    HUMAN_GOTO_ARRIVAL_TOLERANCE_M,
    NAVIGATION_CONTROLLER_SPACING_M,
    NAVIGATION_CONTROLLER_REACH_TOLERANCE_M,
    NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD,
    NAVIGATION_SEGMENT_HORIZON_M,
    NAVIGATION_SEGMENT_MAX_TURN_RAD,
    NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
    ROBOT_HUMAN_CENTER_CLEARANCE_M,
)


# Expressed as a fraction of the frame so the admission survives a resolution
# change; the denominator is the 720p frame the 1200 px figure was measured on,
# not the resolution the ego camera renders at today.
_GOLDEN_OBJECT_MIN_PROJECTED_AREA_RATIO = 1200.0 / (1280.0 * 720.0)
_G1_ROTATION_CONTROLLER_REVISION = "bang-bang-window60-v1"
_G1_GRASP_GROUNDING_REVISION = "certification-standing-cached-v1"
_G1_GT_AABB_DEPTH_PADDING_M = 0.01
_PUBLIC_GRASP_MAX_EXECUTED_REACHES = 20
_G1_MANIPULATION_STANCE_TOLERANCE_M = 0.10
_G1_PLACE_STANCE_TOLERANCE_M = 0.10
# Mid-band standoff so ordinary arrival error stays inside [0.63, 1.0].
_HANDOVER_STANDOFF_M = 0.5 * (HANDOVER_MIN_DISTANCE_M + HANDOVER_MAX_DISTANCE_M)
_HANDOVER_STANDOFF_TOLERANCE_M = 0.10
_HANDOVER_STANDOFF_SAMPLE_SLACK_M = 0.05
_G1_PRECISE_MIN_SPEED_MPS = 0.04
_G1_PRECISE_MAX_SPEED_MPS = 0.25
# RGB-D support is matched to approximate evaluator volumes.  Small rigid
# objects need a tight tolerance (5 cm would engulf most of this bottle's
# surrounding tray), while the canonical Human capsule needs room for the
# skinned mesh/root-pose offset during the first AnimPeople frame.
_GOLDEN_OBJECT_DEPTH_SUPPORT_TOLERANCE_M = 0.01
_PLACEMENT_SUCCESS_TYPES = frozenset(
    {
        "object_supported_by",
        "object_on_support",
        "object_placed",
    }
)


def _diagnostic_check_types(episode: Any) -> set[str]:
    success = getattr(episode, "diagnostics", None)
    return {
        str(getattr(check, "type", "") or "")
        for check in getattr(success, "checks", None) or ()
    }


def episode_success_requires_retained_object(episode: Any) -> bool:
    """True when the episode succeeds only if the object is still held.

    Hold-and-transport validation authors a destination pose for the
    loaded route but forbids placement. Golden physical-task-only
    certification has no episode.diagnostics overlay and still certifies place.
    """

    types = _diagnostic_check_types(episode)
    return "robot_holding_object" in types and not (types & _PLACEMENT_SUCCESS_TYPES)


def episode_success_requires_handover(episode: Any) -> bool:
    """True when success is an accepted handover, not a terminal hold."""

    return "handover_accepted" in _diagnostic_check_types(episode)


def _reexpress_grasp_pose_into_camera(
    candidate: Any,
    camera_now_from_camera_cached: np.ndarray,
) -> Any:
    from tarsphere.infra.manipulation.graspgenx import GraspPoseCandidate

    cam_now_from_gripper = np.asarray(
        camera_now_from_camera_cached, dtype=np.float64
    ) @ np.asarray(candidate.camera_from_gripper, dtype=np.float64)
    return GraspPoseCandidate(
        confidence=float(candidate.confidence),
        camera_from_gripper=tuple(
            tuple(float(value) for value in row) for row in cam_now_from_gripper
        ),
    )


class GoldenOracleAgent:
    """Execute an authored task-profile route with GT A* navigation."""

    name = "golden_oracle"
    certification_only = True

    def __init__(
        self,
        *,
        controller_spacing_m: float = NAVIGATION_CONTROLLER_SPACING_M,
    ) -> None:
        self.controller_spacing_m = float(controller_spacing_m)
        self.episode: Any = None
        self._episode_raw: dict[str, Any] = {}
        self._occupancy: Any = None
        self.planner: Planner | None = None
        self.phases: list[dict[str, Any]] = []
        self.tools: Any = None
        self.phase_index = 0
        self._nav_started = False
        self._nav_path: list[list[float]] = []
        self._nav_entry_target_yaw: float | None = None
        self._latest_feedback: dict[str, Any] = {}
        self._human_specs: list[Any] = []
        self._current_observation: dict[str, Any] = {}
        self._decision_requirements: Any = None
        self._oracle_world_snapshot: dict[str, Any] = {}
        self.certification_stage = "C"
        self._completed_certification_gates: set[str] = set()
        self._pending_certification_gate: str | None = None
        self._target_observation_evidence: list[dict[str, Any]] = []
        self._grasp_reach_evidence: list[dict[str, Any]] = []
        self._physical_task_only = False
        self._action_step = 0

    def configure_certification_stage(self, stage: str) -> None:
        """Select the cumulative H1 certification gate executed by this run."""
        normalized = str(stage).strip().upper()
        if normalized not in {"A", "B", "C"}:
            raise ValueError(
                f"Golden Oracle certification stage must be A, B, or C; got {stage!r}."
            )
        self.certification_stage = normalized

    def configure_physical_task_only(self, enabled: bool = True) -> None:
        """Certify only the shared navigate–grasp–carry–place task.

        This mode isolates embodiment execution from Perception, Planning,
        Interaction, and Adaptation opportunities in the surrounding episode.
        It retains the episode's exact physical task, routes, target, and
        destination.
        """
        self._physical_task_only = bool(enabled)

    def attach_public_tools(self, tools: Any) -> None:
        """Bind the evaluated tool registry for hold-and-transport grasps."""

        if tools is None:
            raise ValueError("Public grasp tools must be a registry.")
        self.tools = tools

    def configure_episode(self, episode: Any) -> None:
        """Compile an episode-specific oracle policy before runner reset."""
        raw = episode.model_dump(mode="python")
        self.episode = episode
        self._episode_raw = raw
        occupancy = load_episode_occupancy(episode)
        self._occupancy = occupancy
        self.planner = Planner(
            occupancy,
            backend="numpy",
            simplify=True,
            time_limit_s=30.0,
            # The one navigation planner represents the physical G1 base,
            # not a point robot.  This footprint is identical before and
            # after grasp; it does not select a different motion backend.
            clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
            cache_static_nav=True,
        )
        compiled_phases = (
            self._build_physical_task_phases(episode)
            if self._physical_task_only
            else self._build_phases(episode)
        )
        self.phases = self._phases_for_certification(
            compiled_phases, self.certification_stage
        )
        self.reset()

    def configure_grasp_approach(self, episode: Any) -> None:
        """Bind occupancy and compile only the walk to grasp stance.

        This is the physical-task prefix used by certification
        (authored ``navigate_to_target`` then ``precise_approach`` /
        ``orient``). It does not compile grasp, place, or social phases.
        """
        raw = episode.model_dump(mode="python")
        self.episode = episode
        self._episode_raw = raw
        occupancy = load_episode_occupancy(episode)
        self._occupancy = occupancy
        self.planner = Planner(
            occupancy,
            backend="numpy",
            simplify=True,
            time_limit_s=30.0,
            clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
            cache_static_nav=True,
        )
        self.phases = self.build_grasp_approach_phases(episode)
        self.reset()

    def configure_oracle_grasp(self, episode: Any, tools: Any) -> None:
        """Bind occupancy and compile the walk plus public GT AABB grasp.

        Same locomotion prefix as ``configure_grasp_approach``, then the
        public ``grasp_reach`` phase. ``_public_grasp_object`` tilts the
        ego camera and proposes GraspGenX poses from the frozen episode
        AABB. It does not run SAM3 phrase grounding and does not crouch.
        """
        self.configure_grasp_approach(episode)
        self.attach_public_tools(tools)
        self.phases = self.build_oracle_grasp_phases(episode)

    def configure_episode_humans(self, humans: list[Any]) -> None:
        self._human_specs = list(humans)

    def update_world_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Receive evaluator-private actor poses for Golden certification only."""
        self._oracle_world_snapshot = dict(snapshot or {})

    def reset(self) -> None:
        self.phase_index = 0
        self._action_step = 0
        self._nav_started = False
        self._nav_path = []
        self._nav_entry_target_yaw = None
        self._latest_feedback = {}
        self._current_observation = {}
        self._decision_requirements = None
        self._oracle_world_snapshot = {}
        self._completed_certification_gates.clear()
        self._pending_certification_gate = None
        self._target_observation_evidence.clear()
        self._grasp_reach_evidence.clear()

    def certification_status(self) -> dict[str, Any]:
        return {
            "stage": self.certification_stage,
            "completed_gates": sorted(self._completed_certification_gates),
            "target_observation_evidence": list(self._target_observation_evidence),
            "grasp_reach_evidence": list(self._grasp_reach_evidence),
        }

    @staticmethod
    def _phases_for_certification(
        phases: list[dict[str, Any]], stage: str
    ) -> list[dict[str, Any]]:
        if stage == "C":
            return phases
        stop_index = next(
            (
                index
                for index, phase in enumerate(phases)
                if str(phase.get("certification_gate") or "") == stage
            ),
            None,
        )
        if stop_index is None:
            raise ValueError(
                f"Golden Oracle phase graph does not define certification gate {stage}."
            )
        return [
            *phases[: stop_index + 1],
            {
                "kind": "report",
                "reason": f"Golden Oracle certification gate {stage} complete.",
            },
        ]

    @staticmethod
    def _actor_exclusion_zones(
        episode: dict[str, Any],
    ) -> list[tuple[float, float, float]]:
        """Compile each Human's transport-time position into a safety disk.

        Golden scripts complete authored Human motion before object transport
        begins.  The relevant obstacle is therefore the terminal
        route position, not the already-vacated swept corridor.
        """
        world = episode.get("world_knowledge") or {}
        anchors = world.get("anchors") or {}
        positions_by_id: dict[str, list[float]] = {}
        for human in episode.get("humans") or []:
            human_id = str(human.get("id") or "")
            start = human.get("start_position") or []
            if human_id and len(start) >= 2:
                positions_by_id[human_id] = [
                    float(start[0]),
                    float(start[1]),
                ]
        for milestone in episode.get("milestones") or []:
            for action in milestone.get("human_actions") or []:
                if str(action.get("command") or "") != "goto":
                    continue
                human_id = str(action.get("human") or "")
                anchor = anchors.get(str(action.get("target") or ""))
                center = anchor.get("center") if isinstance(anchor, dict) else None
                if (
                    human_id not in positions_by_id
                    or not isinstance(center, (list, tuple))
                    or len(center) < 2
                ):
                    raise ValueError(
                        "Compiled Human GoTo must resolve to a declared "
                        f"anchor: {action!r}"
                    )
                positions_by_id[human_id] = [
                    float(center[0]),
                    float(center[1]),
                ]

        return [
            (
                position[0],
                position[1],
                ROBOT_HUMAN_CENTER_CLEARANCE_M,
            )
            for position in positions_by_id.values()
        ]

    def update_info(self, info: dict[str, Any]) -> None:
        self._latest_feedback = dict(info)
        executed = bool(info.get("executed", False))
        if self.phase_index < len(self.phases):
            phase = self.phases[self.phase_index]
            if (
                str(phase.get("kind") or "") == "grasp_reach"
                and phase.get("public_skill")
                and phase.get("_awaiting_result")
            ):
                phase["_awaiting_result"] = False
                if executed:
                    gate = str(phase.get("certification_gate") or "")
                    self.phase_index += 1
                    if gate:
                        self._completed_certification_gates.add(gate)
                    return
                failure = info.get("grasp_failure") or {}
                failure_code = str(failure.get("code") or "")
                grounding = str(phase.get("_active_grounding") or "")
                if failure_code == "support_unstable":
                    raise RuntimeError(
                        "The authored grasp target left its support before "
                        "the hand closed on it. Retrying cannot recover a "
                        "target that is no longer where the episode placed "
                        f"it: {info.get('feedback') or failure}"
                    )
                if failure_code == "reach_envelope":
                    raise RuntimeError(
                        "Public grasp_object exhausted GT AABB grounding with "
                        "a reach-envelope failure. The golden oracle does not "
                        "fall back to SAM3 phrase grounding: "
                        f"{info.get('feedback') or failure}"
                    )
                attempts = int(phase.get("_public_attempts") or 0) + 1
                phase["_public_attempts"] = attempts
                print(
                    "[G1-GRASP] public grasp executed=False; retrying the same "
                    f"grounding {grounding!r} (attempt {attempts}).",
                    flush=True,
                )
                if attempts >= _PUBLIC_GRASP_MAX_EXECUTED_REACHES:
                    raise RuntimeError(
                        "Public grasp_object did not retain the target after "
                        f"{attempts} executed reaches from the authored stance."
                    )
                return
        if self._pending_certification_gate is not None:
            gate = self._pending_certification_gate
            self._pending_certification_gate = None
            if not executed:
                raise RuntimeError(
                    f"Golden Oracle certification gate {gate} action was not "
                    "executed by the environment."
                )
            self._completed_certification_gates.add(gate)
        self._credit_completed_navigation_gate(info, executed=executed)

    def _credit_completed_navigation_gate(
        self,
        info: dict[str, Any],
        *,
        executed: bool,
    ) -> None:
        # Navigate gates are marked when the next act() sees arrival. Success
        # diagnostic checks can end the episode on that arriving action, so credit
        # the gate from execution feedback when the goal is already reached.
        if not executed or self.phase_index >= len(self.phases):
            return
        phase = self.phases[self.phase_index]
        if str(phase.get("kind") or "") != "navigate":
            return
        gate = str(phase.get("certification_gate") or "")
        if not gate or not self._nav_started:
            return
        if not self._navigation_phase_reached_goal(phase, info):
            return
        self._completed_certification_gates.add(gate)

    def _navigation_phase_reached_goal(
        self,
        phase: dict[str, Any],
        info: dict[str, Any],
    ) -> bool:
        pose = info.get("pose")
        if not isinstance(pose, dict):
            raise RuntimeError(
                "Golden Oracle cannot credit a navigation certification gate "
                "without the post-action robot pose."
            )
        position = pose.get("position") or []
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise RuntimeError(
                "Golden Oracle requires robot proprioceptive XY pose to "
                "credit a navigation certification gate."
            )
        goal = [float(phase["goal"][0]), float(phase["goal"][1])]
        tolerance = float(
            phase.get(
                "tolerance_m",
                NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
            )
        )
        return (
            math.dist(
                [float(position[0]), float(position[1])],
                goal,
            )
            <= tolerance
        )

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        action_space: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Publish one GT policy decision and its current-frame provenance.

        Like the harness reference, the authored Oracle consumes only this
        observation, has no queued action block, and makes no model API call.
        Its decision still counts against the diagnostic policy-step budget.
        """
        decision = self._select_action(observation, instruction, action_space, **kwargs)
        decision.update(
            {
                "action_step": self._action_step,
                "policy_step": self._action_step,
                "policy_inference": True,
                "model_visible_action_steps": [self._action_step],
                "model_calls": [],
                "reference_policy": self.name,
                "diagnostic_only": True,
            }
        )
        self._action_step += 1
        return decision

    def _select_action(
        self,
        observation: dict[str, Any],
        instruction: str,
        action_space: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del instruction, action_space
        self._current_observation = dict(observation or {})
        self._decision_requirements = kwargs.get("decision_requirements")
        self._decision_step = kwargs.get("decision_step")
        self._replay_log = kwargs.get("replay_log")
        if self.episode is None or self.planner is None:
            raise RuntimeError("GoldenOracleAgent.configure_episode() was not called.")

        # Advancing a completed phase and emitting its successor in the same
        # decision avoids artificial wait steps in the certification count.
        while self.phase_index < len(self.phases):
            phase = self.phases[self.phase_index]
            kind = str(phase["kind"])
            if kind == "navigate":
                decision = self._navigate(phase, observation)
                if decision is None:
                    self._advance_phase()
                    continue
                return decision
            if kind == "wait_for_human":
                decision = self._wait_for_human(phase)
                if decision is None:
                    self._advance_phase()
                    continue
                return decision
            if kind == "precise_approach":
                decision = self._precise_approach(phase, observation)
                if decision is None:
                    self._advance_phase()
                    continue
                return decision
            if kind == "back_away":
                decision = self._back_away(phase)
                if decision is None:
                    self._advance_phase()
                    continue
                return decision
            if kind == "orient":
                decision = self._orient(phase, observation)
                if decision is None:
                    self._advance_phase()
                    continue
                return decision
            if kind == "target_observation":
                self._record_target_observation(phase)
                self._advance_phase()
                continue
            if kind == "arm_pose":
                self._advance_phase(defer_gate=True)
                return self._arm_pose_decision(phase, observation)
            if kind == "body_posture":
                self._advance_phase()
                action = str(phase["action"])
                return self._tool_decision(
                    action,
                    (
                        {"pelvis_height_m": float(phase["pelvis_height_m"])}
                        if action == "crouch"
                        else {}
                    ),
                    str(phase["reason"]),
                )
            if kind == "grasp_reach":
                if phase.get("public_skill"):
                    return self._public_grasp_object(phase)
                return self._grasp_reach(phase)
            if kind == "communicate":
                self._advance_phase()
                return self._communication_decision(phase)
            if kind == "handover":
                self._advance_phase()
                return self._tool_decision(
                    "handover_object",
                    {"recipient": str(phase["target"])},
                    str(
                        phase.get("reason")
                        or f"Offer the held object to {phase['target']}."
                    ),
                )
            if kind == "release":
                self._advance_phase(defer_gate=True)
                return self._tool_decision(
                    "place_object",
                    {"hand": phase.get("hand", "right"), "object_id": phase["target"]},
                    f"Oracle physical release of {phase['target']} at the authored receptacle",
                    upper_command=phase.get("upper_command"),
                    place_ik=phase.get("place_ik"),
                )
            if kind == "wait":
                remaining = int(phase.get("remaining", 1))
                if remaining <= 0:
                    self._advance_phase()
                    continue
                phase["remaining"] = remaining - 1
                return self._tool_decision(
                    "wait", {}, phase.get("reason", "oracle settle")
                )
            if kind == "report":
                self._advance_phase()
                return self._tool_decision(
                    "report_done",
                    {},
                    "Authored physical route and interaction chain complete.",
                )
            raise ValueError(f"Unknown Golden oracle phase: {phase!r}")

        return self._tool_decision("report_done", {}, "Oracle phase list exhausted.")

    def _navigate(
        self, phase: dict[str, Any], observation: dict[str, Any]
    ) -> dict[str, Any] | None:
        position = (observation.get("pose") or {}).get("position") or []
        if len(position) < 2:
            raise RuntimeError("Golden oracle requires robot proprioceptive XY pose.")
        current = [float(position[0]), float(position[1])]
        goal = [float(phase["goal"][0]), float(phase["goal"][1])]
        tolerance = float(
            phase.get(
                "tolerance_m",
                NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
            )
        )
        distance = math.dist(current, goal)
        controller = ((observation.get("proprio") or {}).get("controllers") or {}).get(
            "move_along_path"
        ) or {}
        controller_finished = bool(controller.get("finished", False))
        segment_finished = bool(
            self._nav_started
            and self._nav_path
            and math.dist(current, self._nav_path[-1][:2])
            <= NAVIGATION_CONTROLLER_REACH_TOLERANCE_M
        )
        if self._nav_started and distance <= tolerance:
            return None
        # Explicit controller completion may request a fresh path from the
        # same planner/backend. Physical stall detection is owned solely by
        # the rollout ProgressWatchdog.
        if self._nav_started and (controller_finished or segment_finished):
            self._nav_started = False
            self._nav_path = []
            self._nav_entry_target_yaw = None

        path_entry_heading_error: float | None = None
        path_entry_target_yaw: float | None = None
        if not self._nav_started and not self._nav_path:
            yaw = float((observation.get("pose") or {}).get("yaw_rad") or 0.0)
            active_planner = (
                self._live_actor_planner()
                if phase.get("live_actor_clearance", False)
                else self.planner
            )
            if active_planner is None:
                raise RuntimeError("Golden oracle planner was not configured.")
            trace_plan = os.environ.get(
                "TARSPHERE_ROLLOUT_MEMORY_DIAGNOSTICS", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if trace_plan:
                print(
                    "[ORACLE-PLAN] begin "
                    f"phase={phase.get('label')!r} current={current!r} "
                    f'goal={goal!r} backend="move_along_path"',
                    flush=True,
                )
            plan_started = time.monotonic()
            plan = active_planner.plan(
                PlanInput(
                    start_xy=(current[0], current[1]),
                    start_yaw_rad=yaw,
                    goal_xy=(goal[0], goal[1]),
                )
            )
            if trace_plan:
                print(
                    "[ORACLE-PLAN] end "
                    f"phase={phase.get('label')!r} status={plan.status!r} "
                    f"elapsed_s={time.monotonic() - plan_started:.4f} "
                    f"waypoints={len(plan.waypoints)}",
                    flush=True,
                )
            resolved = list(plan.goal_xy or goal)
            if (
                plan.status == "already_there"
                or math.dist(current, resolved) <= tolerance
            ):
                return None
            if plan.status != "ok" or not plan.waypoints:
                raise RuntimeError(
                    f"Golden oracle cannot plan {phase.get('label')}: "
                    f"status={plan.status}, current={current}, goal={goal}"
                )
            sampled = self._resample_preserving_vertices(
                list(plan.waypoints),
                start_xy=current,
                spacing_m=self.controller_spacing_m,
            )
            # The stateful controller owns this stable prefix until it reports
            # completion.  Replan only at that boundary, never every decision.
            # A single universal horizon prevents a long controller command
            # from carrying stale waypoint state through several room turns.
            sampled = self._path_prefix(
                current,
                sampled,
                max_segment_m=NAVIGATION_SEGMENT_HORIZON_M,
            )

            heading_target = next(
                (point for point in sampled if math.dist(current, point[:2]) >= 0.2),
                sampled[-1],
            )
            desired_yaw = math.atan2(
                float(heading_target[1]) - current[1],
                float(heading_target[0]) - current[0],
            )
            # Planning and path-entry alignment are one navigation phase.
            # Lock the collision-checked path and its first-segment heading
            # before rotating; small root drift during the official G1 yaw
            # controller must not silently select a different A* branch on
            # every decision. Only explicit controller completion may discard
            # this reference and request a new plan.
            self._nav_path = [
                [float(point[0]), float(point[1]), 0.0] for point in sampled
            ]
            self._nav_entry_target_yaw = float(desired_yaw)

        if not self._nav_started:
            if not self._nav_path or self._nav_entry_target_yaw is None:
                raise RuntimeError(
                    "Golden oracle path-entry alignment has no locked path."
                )
            yaw = float((observation.get("pose") or {}).get("yaw_rad") or 0.0)
            desired_yaw = float(self._nav_entry_target_yaw)
            heading_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
            path_entry_heading_error = float(heading_error)
            path_entry_target_yaw = float(desired_yaw)
            if abs(heading_error) > NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD:
                rotation = self._rotation_decision(
                    goal_yaw=desired_yaw,
                    current_yaw=yaw,
                    label=f"{phase.get('label') or 'semantic stop'} path entry",
                    reason="Align G1 with the planned path before translation.",
                    rotation_objective="path_entry",
                )
                if rotation is not None:
                    return rotation
            # A long in-place turn can translate the official G1 root by
            # several centimetres.  Preserve the planned A* branch and its
            # locked entry heading, but do not ask MoveToPoint to chase a
            # waypoint which the live root has already consumed or passed.
            # This is a geometric prefix trim, not a new planner call.
            self._nav_path = self._trim_locked_path_prefix(
                current,
                self._nav_path,
                entry_yaw=desired_yaw,
            )
            self._nav_started = True

        label = str(phase.get("label") or "semantic stop")
        envelope = self._base_envelope()
        envelope["actions"] = [
            {
                "name": "navigation",
                "args": {"goal_xy": goal, "semantic_stop": label},
            }
        ]
        controller_metadata = {
            "skill_action": {
                "action": "follow_path",
                "path_points": self._nav_path,
            },
            "oracle_semantic_stop": label,
            "goal_distance_m": float(distance),
        }
        if path_entry_heading_error is not None:
            controller_metadata["path_entry_heading_error_rad"] = (
                path_entry_heading_error
            )
            controller_metadata["path_entry_target_yaw_rad"] = path_entry_target_yaw
        return {
            "action": "follow_path",
            "thinking": f"Golden oracle follows GT A* path to {label}.",
            "controller_action": {"move_along_path": [self._nav_path]},
            "controller_backend": "move_along_path",
            "controller_metadata": controller_metadata,
            "envelope": envelope,
            "call_name": "navigation",
        }

    @staticmethod
    def _trim_locked_path_prefix(
        current: list[float],
        path: list[list[float]],
        *,
        entry_yaw: float,
        consumed_radius_m: float = 0.25,
        minimum_forward_progress_m: float = 0.10,
    ) -> list[list[float]]:
        """Reconnect a drifted live root to the same locked path branch.

        The official path controller treats its first point as a strict
        target.  After a large entry rotation, that point may lie behind the
        robot even though the remaining path is still valid.  Drop only the
        consumed prefix while retaining at least the terminal waypoint.
        """
        if not path:
            raise RuntimeError("Locked navigation path must not be empty.")
        first_kept = 0
        heading_x = math.cos(float(entry_yaw))
        heading_y = math.sin(float(entry_yaw))
        while first_kept < len(path) - 1:
            point = path[first_kept]
            dx = float(point[0]) - float(current[0])
            dy = float(point[1]) - float(current[1])
            distance = math.hypot(dx, dy)
            forward_progress = dx * heading_x + dy * heading_y
            if distance > float(consumed_radius_m) and forward_progress > float(
                minimum_forward_progress_m
            ):
                break
            first_kept += 1
        return [list(point) for point in path[first_kept:]]

    @staticmethod
    def _path_prefix(
        current: list[float],
        sampled: list[list[float]],
        *,
        max_segment_m: float = NAVIGATION_SEGMENT_HORIZON_M,
        max_turn_rad: float = NAVIGATION_SEGMENT_MAX_TURN_RAD,
        consumed_radius_m: float = NAVIGATION_CONTROLLER_REACH_TOLERANCE_M,
    ) -> list[list[float]]:
        """Return one bounded, approximately straight controller segment.

        The stateful controller consumes this stable prefix without replanning
        on every decision.  A semantic corner ends the prefix so the next
        decision can observe the live body, plan the next segment, and align
        its entry heading before translating again.
        """
        if not sampled:
            raise RuntimeError("Segmented navigation needs a non-empty path.")
        remaining = [list(point) for point in sampled]
        # A* commonly retains its snapped start cell.  It is not a meaningful
        # controller segment once the live root is already inside the shared
        # completion radius; keeping it as the lone point before a corner
        # creates an endlessly re-planned zero-length prefix.
        while len(remaining) > 1 and math.dist(current, remaining[0][:2]) <= float(
            consumed_radius_m
        ):
            remaining.pop(0)
        chosen: list[list[float]] = []
        travelled = 0.0
        previous = [float(current[0]), float(current[1])]
        previous_heading: float | None = None
        for point in remaining:
            segment = math.dist(previous, point[:2])
            if segment <= 1e-9:
                continue
            heading = math.atan2(
                float(point[1]) - previous[1],
                float(point[0]) - previous[0],
            )
            if previous_heading is not None:
                turn = abs(
                    (heading - previous_heading + math.pi) % (2.0 * math.pi) - math.pi
                )
                if chosen and turn > float(max_turn_rad):
                    break
            if chosen and travelled + segment > float(max_segment_m) + 1e-9:
                break
            chosen.append([float(point[0]), float(point[1])])
            travelled += segment
            previous = chosen[-1]
            previous_heading = heading
        if not chosen:
            raise RuntimeError("Segmented navigation could not select a path prefix.")
        return chosen

    def _live_actor_planner(self) -> Planner:
        if self._occupancy is None:
            raise RuntimeError("Golden oracle occupancy was not configured.")
        humans = self._oracle_world_snapshot.get("humans") or {}
        zones: list[tuple[float, float, float]] = []
        if isinstance(humans, dict):
            for state in humans.values():
                position = state.get("position") if isinstance(state, dict) else None
                if isinstance(position, (list, tuple)) and len(position) >= 2:
                    zones.append(
                        (
                            float(position[0]),
                            float(position[1]),
                            ROBOT_HUMAN_CENTER_CLEARANCE_M,
                        )
                    )
        if not zones:
            zones = self._actor_exclusion_zones(self._episode_raw)
        return Planner(
            self._occupancy,
            backend="numpy",
            simplify=True,
            time_limit_s=30.0,
            clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
            world_exclusion_zones=zones,
        )

    @staticmethod
    def _resample_preserving_vertices(
        path: list[list[float]],
        *,
        start_xy: list[float],
        spacing_m: float,
    ) -> list[list[float]]:
        """Subdivide long safe edges without deleting certified vertices."""
        if not path:
            return []
        spacing = max(float(spacing_m), 1e-6)
        sampled: list[list[float]] = []
        previous = [float(start_xy[0]), float(start_xy[1])]
        for vertex in path:
            target = [float(vertex[0]), float(vertex[1])]
            length = math.dist(previous, target)
            subdivisions = max(1, int(math.ceil(length / spacing)))
            for index in range(1, subdivisions + 1):
                ratio = index / subdivisions
                point = [
                    previous[0] + ratio * (target[0] - previous[0]),
                    previous[1] + ratio * (target[1] - previous[1]),
                ]
                if not sampled or math.dist(sampled[-1], point) > 1e-9:
                    sampled.append(point)
            previous = target
        return sampled

    def _precise_approach(
        self, phase: dict[str, Any], observation: dict[str, Any]
    ) -> dict[str, Any] | None:
        position = (observation.get("pose") or {}).get("position") or []
        if len(position) < 2:
            raise RuntimeError(
                "Golden oracle requires robot pose for precise approach."
            )
        current = [float(position[0]), float(position[1])]
        goal = [float(phase["goal"][0]), float(phase["goal"][1])]
        tolerance = float(phase.get("tolerance_m", 0.12))
        distance = math.dist(current, goal)
        completion_place_ik = phase.get("completion_place_ik")
        preserve_heading = bool(phase.get("preserve_heading", False))
        placement_residual = False
        if isinstance(completion_place_ik, dict) and (
            distance <= tolerance
            or bool(phase.get("placement_envelope_entered", False))
        ):
            # Reaching the authored stance transfers control from navigation
            # to the live carried-hand placement geometry.  Keep that transfer
            # latched: the official gait can drift a few centimetres while
            # rotating under payload, and falling back to the nominal stance
            # would make the two controllers reverse each other at the
            # tolerance boundary.
            phase["placement_envelope_entered"] = True
            placement = self._live_g1_place_reachability(
                completion_place_ik, current_xy=current
            )
            phase["last_place_reachability"] = placement
            current_yaw = float((observation.get("pose") or {}).get("yaw_rad") or 0.0)
            semantic_error = (
                float(placement["semantic_yaw_rad"]) - current_yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
            max_facing_error = math.radians(
                float(completion_place_ik.get("max_facing_error_deg", 30.0))
            )
            if placement["reachable"] and abs(semantic_error) <= max_facing_error:
                # The same certified right-hand IK compiles the release action.
                # Once it certifies the carried hand and semantic facing, the
                # stance phase is complete.  Any compliant-controller payload
                # residual belongs to the executor's bounded live-bbox
                # placement loop; duplicating that closure here creates a
                # second, oscillatory body servo and makes grasp geometry alter
                # navigation completion.
                return None
            else:
                desired_yaw = float(placement["recommended_yaw_rad"])
                yaw_delta = (desired_yaw - current_yaw + math.pi) % (
                    2.0 * math.pi
                ) - math.pi
                if abs(yaw_delta) > float(
                    phase.get("placement_yaw_tolerance_rad", 0.08)
                ):
                    return self._rotation_decision(
                        goal_yaw=desired_yaw,
                        current_yaw=current_yaw,
                        label=str(phase.get("label") or "reachable placement envelope"),
                        reason=(
                            "Align the carried hand with the live compensated "
                            "placement target before release IK."
                        ),
                        rotation_objective="placement_target_facing",
                    )

                # A residual radial miss after compensated orientation is
                # closed with one bounded base step.  The completion predicate
                # remains carried-hand IK, never the nominal base coordinate.
                base_translation = np.asarray(
                    placement["base_translation_world"],
                    dtype=np.float64,
                ).reshape(-1)
                if base_translation.size != 3 or not np.all(
                    np.isfinite(base_translation)
                ):
                    raise ValueError(
                        "Live placement base translation must contain three "
                        "finite world-frame coordinates."
                    )
                horizontal_translation = base_translation[:2]
                horizontal_distance = float(np.linalg.norm(horizontal_translation))
                if horizontal_distance <= 1e-4:
                    raise RuntimeError(
                        "Live placement remains outside the arm envelope but "
                        "has no correctable horizontal base residual: "
                        f"position_error_m={placement['position_error_m']!r}, "
                        f"base_translation_world={base_translation.tolist()!r}."
                    )
                reach_gap = min(0.04, horizontal_distance)
                goal = [
                    current[0]
                    + reach_gap
                    * float(horizontal_translation[0])
                    / horizontal_distance,
                    current[1]
                    + reach_gap
                    * float(horizontal_translation[1])
                    / horizontal_distance,
                ]
                distance = reach_gap
                preserve_heading = True
                placement_residual = True
        elif distance <= tolerance:
            return None

        label = str(phase.get("label") or "precise manipulation stance")
        attempts = int(phase.get("attempts", 0)) + 1
        phase["attempts"] = attempts
        if attempts > 80:
            raise RuntimeError(
                f"Precise stance servo did not converge for {label!r}: "
                f"current={current!r}, goal={goal!r}, distance_m={distance:.4f}."
            )
        yaw = float((observation.get("pose") or {}).get("yaw_rad") or 0.0)
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        # Lock one travel frame when this stance begins.  The official G1
        # turns with a small but real XY drift; recomputing atan2(goal-current)
        # after every turn makes the requested yaw move with that drift and
        # can send the robot around the stance in a spiral.  The fixed frame
        # owns travel only.  A following ``orient`` phase independently faces
        # the manipulation target after the position has converged.
        if preserve_heading:
            goal_yaw = yaw
        else:
            travel_yaw = phase.get("travel_yaw_rad")
            if travel_yaw is None:
                travel_yaw = math.atan2(dy, dx)
                phase["travel_yaw_rad"] = float(travel_yaw)
            goal_yaw = float(travel_yaw)
        heading_error = (goal_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        if (
            not preserve_heading
            and abs(heading_error) > NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD
        ):
            return self._rotation_decision(
                goal_yaw=goal_yaw,
                current_yaw=yaw,
                label=label,
                reason=(
                    "Align G1 with the remaining manipulation-stance "
                    "displacement before walking forward."
                ),
                rotation_objective="manipulation_stance_entry",
            )
        speed = min(
            _G1_PRECISE_MAX_SPEED_MPS,
            max(_G1_PRECISE_MIN_SPEED_MPS, distance),
        )
        # Close the fixed world-space position in the current body frame.
        # This corrects the bounded XY drift introduced by rotation without
        # changing the locked travel yaw or asking the planner for a new path.
        unit_x = dx / max(distance, 1e-9)
        unit_y = dy / max(distance, 1e-9)
        forward = speed * (math.cos(yaw) * unit_x + math.sin(yaw) * unit_y)
        lateral = speed * (-math.sin(yaw) * unit_x + math.cos(yaw) * unit_y)
        if placement_residual:
            # Placement first aligns the torso with a live, semantically
            # bounded hand target. Preserve that facing while closing the
            # remaining <=4 cm translation; the next decision recompiles this
            # bounded body-frame step from live hand/object geometry.
            servo_mode = "placement_residual"
        elif preserve_heading:
            # After a target-facing turn, close the resulting XY drift in the
            # current body frame without turning away from the target.
            servo_mode = "target_facing_stance_recovery"
        else:
            servo_mode = "locked_travel_frame"
        envelope = self._base_envelope()
        envelope["actions"] = [
            {
                "name": "navigation",
                "args": {"goal_xy": goal, "semantic_stop": label},
            }
        ]
        decision = {
            "action": "precise_approach",
            "thinking": (
                f"Golden oracle closes the measured body-frame XY error to "
                f"{label} with a bounded velocity servo."
            ),
            "controller_action": {
                "move_by_speed": [forward, lateral, 0.0],
            },
            "controller_backend": "move_by_speed",
            "controller_metadata": {
                "skill_action": {"action": "precise_approach", "goal": goal},
                "oracle_semantic_stop": label,
                "precise_stance_servo": {
                    "distance_m": distance,
                    "speed_mps": speed,
                    "forward_mps": forward,
                    "lateral_mps": lateral,
                    "heading_error_rad": heading_error,
                    "mode": servo_mode,
                    "attempt": attempts,
                },
            },
            "envelope": envelope,
            "call_name": "navigation",
        }
        return decision

    def _live_g1_arm_reachability(self, arm_ik: dict[str, Any]) -> dict[str, Any]:
        """Evaluate pre-grasp completion with the certified right-hand IK."""
        offset, linear = self._live_torso_affine()
        target_world = np.asarray(
            arm_ik.get("target_position_world"), dtype=np.float64
        ).reshape(-1)
        if target_world.size != 3 or not np.all(np.isfinite(target_world)):
            raise ValueError("Live grasp target must contain three finite values.")
        target_torso = offset + linear @ target_world
        tolerance = float(arm_ik.get("position_tolerance_m", 0.06))
        from tarsphere.embodiment.dex3 import (
            DEX3_GRASP_CENTER_FROM_PALM_M,
        )
        from tarsphere.infra.manipulation.g1_arm_ik import (
            frontal_grasp_palm_rotation_in_torso,
            solve_hand_pose,
        )

        reference_from_palm = (
            DEX3_GRASP_CENTER_FROM_PALM_M
            if bool(arm_ik.get("align_grasp_center", False))
            else (0.0, 0.0, 0.0)
        )
        target_rotation_torso = (
            frontal_grasp_palm_rotation_in_torso(target_torso)
            if bool(arm_ik.get("align_grasp_center", False))
            else None
        )

        candidate = solve_hand_pose(
            target_torso,
            hand=G1_MANIPULATION_HAND,
            position_tolerance_m=tolerance,
            solver_tolerance_m=min(
                tolerance,
                G1_ARM_IK_SOLVER_TOLERANCE_M,
            ),
            reference_from_palm=reference_from_palm,
            target_rotation_torso=target_rotation_torso,
        )
        selected = candidate if candidate.converged else None
        return {
            "reachable": selected is not None,
            "hand": None if selected is None else selected.hand,
            "target_torso": target_torso.tolist(),
            "tolerance_m": tolerance,
            "best_error_m": float(candidate.position_error_m),
            "errors_m": {G1_MANIPULATION_HAND: float(candidate.position_error_m)},
            "orientation_errors_rad": {
                G1_MANIPULATION_HAND: float(candidate.orientation_error_rad)
            },
        }

    def _live_g1_place_reachability(
        self,
        place_ik: dict[str, Any],
        *,
        current_xy: list[float],
    ) -> dict[str, Any]:
        """Predict the exact carried-hand placement target from live geometry.

        The private manipulation snapshot exposes the same live hand point and
        object AABB consumed by ``G1AttachmentExecutor.placement_hand_target``.
        Reusing its pure geometry function lets the base align the *carried*
        certified right hand before the irreversible release window.
        """
        manipulation = self._oracle_world_snapshot.get("manipulation") or {}
        objects = (
            manipulation.get("objects") if isinstance(manipulation, dict) else None
        )
        object_id = str(place_ik.get("object_id") or "")
        state = objects.get(object_id) if isinstance(objects, dict) else None
        if not isinstance(state, dict) or not bool(state.get("attached", False)):
            raise RuntimeError(
                f"Live placement requires attached geometry for {object_id!r}."
            )
        hand = str(state.get("hand") or "").lower()
        if hand != G1_MANIPULATION_HAND:
            raise RuntimeError(
                f"Live placement is not held by the certified right hand: {hand!r}."
            )
        lower = np.asarray(state.get("bbox_min"), dtype=np.float64).reshape(-1)
        upper = np.asarray(state.get("bbox_max"), dtype=np.float64).reshape(-1)
        hand_world = np.asarray(
            state.get("hand_position_world"), dtype=np.float64
        ).reshape(-1)
        surface = np.asarray(
            place_ik.get("support_surface_world"), dtype=np.float64
        ).reshape(-1)
        if (
            lower.size != 3
            or upper.size != 3
            or hand_world.size != 3
            or surface.size != 3
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or not np.all(np.isfinite(hand_world))
            or not np.all(np.isfinite(surface))
            or np.any(lower > upper)
        ):
            raise ValueError("Live placement geometry is malformed.")

        offset, linear = self._live_torso_affine()

        from tarsphere.infra.manipulation.g1_arm_ik import solve_hand_pose
        from tarsphere.infra.manipulation.g1_attachment import (
            placement_hand_target_from_geometry,
        )

        clearance = G1_PRE_RELEASE_CLEARANCE_M
        target_world = np.asarray(
            placement_hand_target_from_geometry(
                hand_world=hand_world.tolist(),
                object_bbox_world=(lower.tolist(), upper.tolist()),
                support_surface_world=surface.tolist(),
                clearance_m=clearance,
            ),
            dtype=np.float64,
        )
        target_torso = offset + linear @ target_world
        tolerance = G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M
        result = solve_hand_pose(
            target_torso,
            hand=hand,
            position_tolerance_m=tolerance,
        )
        hand_torso = np.asarray(result.hand_position_torso, dtype=np.float64).reshape(
            -1
        )
        if hand_torso.size != 3 or not np.all(np.isfinite(hand_torso)):
            raise RuntimeError(
                "G1 placement IK returned a malformed reachable hand position."
            )
        try:
            base_translation_world = np.linalg.solve(
                linear,
                target_torso - hand_torso,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Live torso transform is singular; placement residual cannot "
                "be mapped into world coordinates."
            ) from exc
        print(
            "[ORACLE-PLACE] live IK envelope: "
            f"hand={hand} converged={bool(result.converged)} "
            f"position_error_m={float(result.position_error_m):.4f} "
            f"tolerance_m={tolerance:.4f} "
            f"target_torso="
            f"{[round(float(value), 4) for value in target_torso]} "
            f"reachable_hand_torso="
            f"{[round(float(value), 4) for value in hand_torso]} "
            f"base_translation_world="
            f"{[round(float(value), 4) for value in base_translation_world]}.",
            flush=True,
        )

        compensated_yaw = math.atan2(
            float(target_world[1]) - float(current_xy[1]),
            float(target_world[0]) - float(current_xy[0]),
        )
        semantic_yaw = math.atan2(
            float(surface[1]) - float(current_xy[1]),
            float(surface[0]) - float(current_xy[0]),
        )
        max_facing_error = math.radians(
            float(place_ik.get("max_facing_error_deg", 30.0))
        )
        relative = (compensated_yaw - semantic_yaw + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        bounded_relative = min(
            max(relative, -max_facing_error + 1e-3),
            max_facing_error - 1e-3,
        )
        recommended_yaw = semantic_yaw + bounded_relative
        return {
            "reachable": bool(result.converged),
            "hand": hand,
            "position_error_m": float(result.position_error_m),
            "tolerance_m": tolerance,
            "target_position_world": target_world.tolist(),
            "target_position_torso": target_torso.tolist(),
            "reachable_hand_position_torso": hand_torso.tolist(),
            "base_translation_world": base_translation_world.tolist(),
            "recommended_yaw_rad": float(recommended_yaw),
            "semantic_yaw_rad": float(semantic_yaw),
        }

    def _live_torso_affine(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the validated private world-to-torso affine transform."""
        robot = self._oracle_world_snapshot.get("robot") or {}
        affine = robot.get("torso_world_to_frame") if isinstance(robot, dict) else None
        if not isinstance(affine, dict):
            raise RuntimeError(
                "Golden manipulation IK requires the live torso frame in "
                "the benchmark-private world snapshot."
            )
        offset = np.asarray(affine.get("offset"), dtype=np.float64).reshape(-1)
        linear = np.asarray(affine.get("linear"), dtype=np.float64)
        if (
            offset.size != 3
            or linear.shape != (3, 3)
            or not np.all(np.isfinite(offset))
            or not np.all(np.isfinite(linear))
        ):
            raise ValueError("Live world-to-torso affine transform is malformed.")
        return offset, linear

    def _back_away(self, phase: dict[str, Any]) -> dict[str, Any] | None:
        """Execute a bounded body-frame retreat before path planning."""
        remaining = int(phase.get("remaining", phase.get("windows", 1)))
        if remaining <= 0:
            return None
        phase["remaining"] = remaining - 1
        speed = abs(float(phase.get("speed_mps", 0.5)))
        label = str(phase.get("label") or "post-manipulation clearance")
        return {
            "action": "precise_approach",
            "thinking": (
                f"Back away from {label} before handing control to the "
                "global path controller."
            ),
            "controller_action": {
                "move_by_speed": [-speed, 0.0, 0.0],
            },
            "controller_backend": "move_by_speed",
            "controller_metadata": {
                "skill_action": {
                    "action": "back_away",
                    "speed_mps": speed,
                    "remaining_windows": remaining - 1,
                },
                "oracle_semantic_stop": label,
            },
            "envelope": self._base_envelope(),
            "call_name": "navigation",
        }

    def _wait_for_human(self, phase: dict[str, Any]) -> dict[str, Any] | None:
        name = str(phase["human"])
        target = [float(phase["target"][0]), float(phase["target"][1])]
        tolerance = HUMAN_GOTO_ARRIVAL_TOLERANCE_M
        character_id = next(
            (
                str(getattr(human, "id", ""))
                for human in self._human_specs
                if str(getattr(human, "name", "")) == name
            ),
            "",
        )
        state = self._human_runtime_state(character_id)
        position = state.get("position")
        animation = str(state.get("animation_state") or "").casefold()
        if (
            isinstance(position, (list, tuple))
            and len(position) >= 2
            and math.dist([float(position[0]), float(position[1])], target) <= tolerance
            and "walk" not in animation
            and "goto" not in animation
            and "run" not in animation
        ):
            return None
        attempts = int(phase.get("attempts", 0)) + 1
        phase["attempts"] = attempts
        if attempts > int(phase.get("max_attempts", 20)):
            raise RuntimeError(
                f"Human {name!r} did not clear the coordinated route: "
                f"target={target!r}, last_state={state!r}."
            )
        return {
            "action": "coordination_wait",
            "thinking": f"Hold position until {name} clears the placement route.",
            "controller_action": {"move_by_speed": [0.0, 0.0, 0.0]},
            "controller_backend": "move_by_speed",
            "controller_metadata": {
                "oracle_coordination_wait": True,
                "target_human": name,
            },
            "envelope": self._base_envelope(),
            "call_name": "wait",
        }

    def _orient(
        self, phase: dict[str, Any], observation: dict[str, Any]
    ) -> dict[str, Any] | None:
        pose = observation.get("pose") or {}
        current_yaw = float(pose.get("yaw_rad") or 0.0)
        current_position = pose.get("position") or []
        stance = phase.get("stance_xy")
        if (
            not isinstance(stance, (list, tuple))
            or len(stance) != 2
            or len(current_position) < 2
        ):
            raise ValueError(
                "Target-facing orientation requires a two-dimensional "
                "stance and a live Robot position."
            )
        stance_distance = math.dist(
            [float(current_position[0]), float(current_position[1])],
            [float(stance[0]), float(stance[1])],
        )
        stance_tolerance = float(phase["stance_tolerance_m"])
        # Position and target-facing are consecutive manipulation gates, not
        # two controllers competing over the same pose. Once the Robot has
        # entered the certified stance envelope, retain that fact while the
        # official gait turns toward the target. Rotation has bounded XY
        # drift; chasing the pre-turn coordinate afterward can only command a
        # target-facing Robot back toward a point behind it.
        stance_admitted = bool(phase.get("stance_envelope_entered", False))
        if not stance_admitted and stance_distance > stance_tolerance:
            recovery = phase.get("stance_recovery")
            if not isinstance(recovery, dict):
                recovery = self._precise_phase(
                    stance,
                    f"{phase.get('label') or 'target-facing'} stance recovery",
                    tolerance_m=stance_tolerance,
                )
                recovery["preserve_heading"] = True
                phase["stance_recovery"] = recovery
            decision = self._precise_approach(recovery, observation)
            if decision is not None:
                return decision
            phase.pop("stance_recovery", None)
            stance_admitted = True
        elif not stance_admitted:
            stance_admitted = True
        if stance_admitted:
            phase["stance_envelope_entered"] = True
        if len(current_position) >= 2 and "target_xy" in phase:
            target_xy = phase["target_xy"]
            goal_yaw = math.atan2(
                float(target_xy[1]) - float(current_position[1]),
                float(target_xy[0]) - float(current_position[0]),
            )
        else:
            goal_yaw = float(phase["yaw_rad"])
        delta = (goal_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(delta) <= float(phase.get("tolerance_rad", 0.12)):
            return None
        # Every rotation window may translate the official G1 root. Rebuild a
        # stance-recovery servo from the next live pose instead of retaining a
        # travel frame that predates that physical drift.
        phase.pop("stance_recovery", None)
        attempts = int(phase.get("attempts", 0)) + 1
        phase["attempts"] = attempts
        if attempts > 80:
            raise RuntimeError(
                "Required G1 orientation did not converge within 80 "
                f"closed-loop windows for {phase.get('label')!r}: "
                f"current_yaw={current_yaw:.4f}, goal_yaw={goal_yaw:.4f}."
            )
        return self._rotation_decision(
            goal_yaw=goal_yaw,
            current_yaw=current_yaw,
            label=str(phase.get("label") or "manipulation target"),
            reason="Turn the official G1 right hand toward the manipulation target.",
            rotation_objective="manipulation_target_facing",
            preferred_direction=phase.get("preferred_rotation_direction"),
            semantic_target=phase.get("semantic_target"),
        )

    def _rotation_decision(
        self,
        *,
        goal_yaw: float,
        current_yaw: float,
        label: str,
        reason: str,
        rotation_objective: str = "heading_alignment",
        preferred_direction: Any = None,
        semantic_target: Any = None,
    ) -> dict[str, Any] | None:
        delta = (float(goal_yaw) - float(current_yaw) + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        # The official locomotion policy absorbs sub-unit yaw commands after
        # settling, while one fixed 60-tick unit-command window produces a
        # bounded turn of roughly the manipulation tolerance.  Use a
        # sampled bang-bang servo: command the robot-profile effective value,
        # then always re-read physical yaw before deciding whether to turn
        # again.  The semantic angular threshold supplies the fine stop rule.
        direction = math.copysign(1.0, float(delta))
        if preferred_direction is not None and abs(float(delta)) > 0.5:
            direction = math.copysign(1.0, float(preferred_direction))
        rotation_speed = direction * G1_ROTATION_POLICY_COMMAND
        print(
            "[ORACLE-ROTATE] "
            f"controller={_G1_ROTATION_CONTROLLER_REVISION} "
            f"label={label!r} current_yaw={float(current_yaw):.4f} "
            f"goal_yaw={float(goal_yaw):.4f} delta={float(delta):.4f} "
            f"command={rotation_speed:.4f}",
            flush=True,
        )
        envelope = self._base_envelope()
        navigation_args: dict[str, Any] = {
            "target_yaw_rad": float(goal_yaw),
            "semantic_stop": label,
        }
        if semantic_target is not None and str(semantic_target).strip():
            navigation_args["target"] = str(semantic_target)
        envelope["actions"] = [
            {
                "name": "navigation",
                "args": navigation_args,
            }
        ]
        return {
            "action": "precise_rotate",
            "target_yaw_rad": goal_yaw,
            "thinking": f"Golden oracle: {reason}",
            "controller_action": {
                "move_by_speed": [0.0, 0.0, rotation_speed],
            },
            "controller_backend": "move_by_speed",
            "controller_metadata": {
                "skill_action": {"action": "rotate_to", "target_yaw_rad": goal_yaw},
                "angular_error_rad": float(delta),
                "angular_speed_command_rad_s": rotation_speed,
                "angular_speed_limit_rad_s": 1.0,
                "rotation_objective": str(rotation_objective),
                "oracle_semantic_stop": label,
            },
            "envelope": envelope,
            "call_name": "navigation",
        }

    def _arm_pose_decision(
        self, phase: dict[str, Any], observation: dict[str, Any]
    ) -> dict[str, Any]:
        action = str(phase["action"])
        hand = str(phase.get("hand") or "right")
        skill_action = {"action": action, "hand": hand}
        if phase.get("target"):
            skill_action["object_id"] = str(phase["target"])
        target_world = phase.get("target_position_world")
        if isinstance(target_world, (list, tuple)) and len(target_world) >= 3:
            decision = {
                "action": action,
                "thinking": str(phase["reason"]),
                "controller_metadata": {
                    "skill_action": skill_action,
                    "g1_arm_ik": {
                        "hand": hand,
                        "target_position_world": [
                            float(target_world[0]),
                            float(target_world[1]),
                            float(target_world[2]),
                        ],
                        "position_tolerance_m": float(
                            phase.get("position_tolerance_m", 0.06)
                        ),
                        "posture": str(phase.get("posture") or "front_right"),
                        "balance_contralateral_arm": bool(
                            phase.get("balance_contralateral_arm", False)
                        ),
                        "max_facing_error_deg": float(
                            phase.get("max_facing_error_deg", 30.0)
                        ),
                        "align_grasp_center": bool(
                            phase.get("align_grasp_center", False)
                        ),
                    },
                    "g1_arm_ik_finish_on_target": bool(
                        phase.get("finish_on_target", False)
                    ),
                    "g1_arm_ik_finish_tolerance_rad": float(
                        phase.get("finish_tolerance_rad", 0.08)
                    ),
                    "g1_pin_base_for_arm_reach": bool(
                        phase.get("pin_base_for_arm_reach", False)
                    ),
                },
                "envelope": self._base_envelope(),
                "call_name": "manipulation",
            }
            if phase.get("internal_controller_action", False):
                # Internal articulated transitions are not part of the
                # benchmark action vocabulary.  Supplying the zero-velocity
                # controller envelope makes that distinction explicit to the
                # runner; Isaac replaces its upper-body component with the
                # certified IK result before advancing physics.
                decision["controller_action"] = {
                    "move_by_speed": [0.0, 0.0, 0.0],
                }
                decision["controller_backend"] = "move_by_speed"
            if phase.get("target"):
                decision["object_id"] = str(phase["target"])
            return decision

        remaining_fraction = phase.get("locomotion_upper_body_remaining_fraction")
        if remaining_fraction is not None:
            fraction = float(remaining_fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    "Intermediate locomotion upper-body fraction must lie "
                    "strictly between zero and one."
                )
            return {
                "action": action,
                "thinking": str(phase["reason"]),
                # Isaac resolves the live grasp-IK command and materializes
                # the interpolated 15-D target before execution.
                "controller_action": {
                    "move_by_speed": [0.0, 0.0, 0.0],
                },
                "controller_backend": "move_by_speed",
                "controller_metadata": {
                    "skill_action": skill_action,
                    "g1_locomotion_upper_body_remaining_fraction": fraction,
                },
                "envelope": self._base_envelope(),
                "call_name": "manipulation",
            }

        upper_command = [float(value) for value in phase["upper_command"]]
        reset_to_locomotion = bool(phase.get("reset_to_locomotion_upper_body", False))
        return {
            "action": action,
            "thinking": str(phase["reason"]),
            "controller_action": {
                "move_by_speed": [0.0, 0.0, 0.0, 0.74, upper_command],
            },
            "controller_backend": "move_by_speed",
            "controller_metadata": {
                "skill_action": skill_action,
                "upper_body_command": upper_command,
                "g1_reset_to_locomotion_upper_body": reset_to_locomotion,
            },
            "envelope": self._base_envelope(),
            "call_name": "manipulation",
        }

    def _public_grasp_object(self, phase: dict[str, Any]) -> dict[str, Any]:
        """Run the evaluated grasp_object tool from the authored stance."""

        if self.tools is None:
            raise RuntimeError(
                "Public grasp_reach requires the evaluated tool registry."
            )
        from tarsphere.harness.certification.grasp_skill_agent import (
            GRASP_CERTIFICATION_CAMERA_TILT_DEG,
        )
        from tarsphere.harness.agent import agent_observation_from_runtime
        from tarsphere.harness.harness import CameraView

        if not phase.get("_camera_tilted"):
            phase["_camera_tilted"] = True
            return self._tool_decision(
                "tilt_camera",
                {"pitch_deg": float(GRASP_CERTIFICATION_CAMERA_TILT_DEG)},
                "Tilt the ego camera onto the low authored target.",
            )

        agent_observation = agent_observation_from_runtime(
            self._current_observation,
            camera_views=(CameraView.EGO,),
        )
        gt_action = self._gt_aabb_grasp_object(
            phase,
            agent_observation,
        )
        phase["_active_grounding"] = "__gt_aabb__"
        phase["_awaiting_result"] = True
        return gt_action

    def _grasp_backend(self) -> Any:
        """Return the GraspGenX client bound to the evaluated grasp tool."""

        if self.tools is None or not hasattr(self.tools, "handler"):
            return None
        handler = self.tools.handler("grasp_object")
        return getattr(handler, "_grasp_backend", None)

    def _gt_aabb_grasp_object(
        self,
        phase: dict[str, Any],
        observation: Any,
    ) -> dict[str, Any]:
        """Propose GraspGenX poses from the frozen episode AABB, not SAM3.

        This is the grasp-asset certificate chain: authored world volume →
        current ego depth mask → GraspGenX → cuRobo. Live USD AABBs in a
        furnished GRScene can swallow the supporting tray and must not be
        the mask. If this volume is not usable the oracle fails; there is
        no other grounding.
        """

        from tarsphere.embodiment.dex3 import G1_MANIPULATION_HAND
        from tarsphere.infra.manipulation.graspgenx import (
            grasp_pose_from_payload,
            grasp_pose_payload,
        )

        grasp_backend = self._grasp_backend()
        if grasp_backend is None:
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: no GraspGenX "
                f"backend ({_G1_GRASP_GROUNDING_REVISION})."
            )
        target_id = str(phase.get("target") or "")
        manipulation = self._oracle_world_snapshot.get("manipulation")
        objects = (
            manipulation.get("objects") if isinstance(manipulation, dict) else None
        )
        target_state = objects.get(target_id) if isinstance(objects, dict) else None
        if not isinstance(target_state, dict):
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: live target "
                f"{target_id!r} is missing from the private snapshot."
            )
        authored = self._authored_pickable_geometry(target_id)
        if authored is None:
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: authored "
                f"target {target_id!r} has no frozen world volume."
            )
        lower, upper, authored_center = authored
        padded_lower = tuple(float(value) for value in lower)
        padded_upper = tuple(float(value) for value in upper)
        ego = observation.views.get("ego")
        if ego is None or ego.depth is None:
            raise RuntimeError("Oracle GT AABB grounding is unavailable: no ego depth.")
        depth = np.squeeze(np.asarray(ego.depth, dtype=np.float32))
        if depth.ndim != 2:
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: depth shape "
                f"{depth.shape!r} is not HxW."
            )
        camera = dict(ego.camera_params or {})
        try:
            intrinsics = G.fov_to_intrinsics(
                float(camera["fov_deg"]),
                int(camera.get("width") or depth.shape[1]),
                int(camera.get("height") or depth.shape[0]),
            )
            world_T_camera = np.asarray(
                camera["world_T_cam"],
                dtype=np.float64,
            )
            capture_id = int(camera["capture_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: camera "
                f"calibration is incomplete: {exc}"
            ) from exc
        if world_T_camera.shape != (4, 4) or not np.all(np.isfinite(world_T_camera)):
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: world_T_cam "
                "is not a finite 4x4 transform."
            )
        lower_arr = np.asarray(padded_lower, dtype=np.float64)
        upper_arr = np.asarray(padded_upper, dtype=np.float64)
        corners_world = np.asarray(
            [
                [x, y, z]
                for x in (lower_arr[0], upper_arr[0])
                for y in (lower_arr[1], upper_arr[1])
                for z in (lower_arr[2], upper_arr[2])
            ],
            dtype=np.float64,
        )
        camera_T_world = np.linalg.inv(world_T_camera)
        corners_camera = (
            np.concatenate(
                [corners_world, np.ones((8, 1), dtype=np.float64)],
                axis=1,
            )
            @ camera_T_world.T
        )[:, :3]
        if np.any(corners_camera[:, 2] <= 0.0):
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: padded target "
                "volume is not fully in front of the ego camera."
            )
        projected = corners_camera @ intrinsics.T
        pixels = projected[:, :2] / projected[:, 2:3]
        height, width = depth.shape
        x1 = max(0, int(np.floor(np.min(pixels[:, 0]))))
        y1 = max(0, int(np.floor(np.min(pixels[:, 1]))))
        x2 = min(width, int(np.ceil(np.max(pixels[:, 0]))) + 1)
        y2 = min(height, int(np.ceil(np.max(pixels[:, 1]))) + 1)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(
                "Oracle GT AABB grounding is unavailable: target projects "
                f"outside the ego frame bbox={[x1, y1, x2, y2]}."
            )
        region_depth = depth[y1:y2, x1:x2]
        region_valid = np.isfinite(region_depth) & (region_depth > 0.0)
        region_rows, region_columns = np.nonzero(region_valid)
        cached = phase.get("_cached_gt_aabb_grasp")
        has_cached_poses = isinstance(cached, dict) and bool(
            cached.get("grasp_pose_candidates")
        )
        live_mask_unusable = False
        if region_rows.size == 0:
            if not has_cached_poses:
                raise RuntimeError(
                    "Oracle GT AABB grounding is unavailable: no valid "
                    "depth inside the projected target box."
                )
            live_mask_unusable = True
            print(
                "[G1-GRASP] oracle GT AABB live mask has no valid depth; "
                f"reusing {len(cached['grasp_pose_candidates'])} cached "
                f"GraspGenX poses into capture_id={capture_id} "
                f"({_G1_GRASP_GROUNDING_REVISION}).",
                flush=True,
            )
        else:
            region_ranges = region_depth[region_valid].astype(
                np.float64,
                copy=False,
            )
            image_rows = region_rows + y1
            image_columns = region_columns + x1
            image_pixels = np.stack(
                [
                    image_columns,
                    image_rows,
                    np.ones_like(image_columns),
                ],
                axis=1,
            ).astype(np.float64, copy=False)
            points_camera = (
                image_pixels @ np.linalg.inv(intrinsics).T
            ) * region_ranges[:, None]
            points_world = G.transform_to_world(points_camera, world_T_camera)
            volume_supported = np.all(
                (points_world >= lower_arr) & (points_world <= upper_arr),
                axis=1,
            )
            target_mask = np.zeros(depth.shape, dtype=bool)
            target_mask[
                image_rows[volume_supported],
                image_columns[volume_supported],
            ] = True
            support_count = int(np.count_nonzero(target_mask))
            if support_count < 100:
                if not has_cached_poses:
                    raise RuntimeError(
                        "Oracle GT AABB grounding is unavailable: only "
                        f"{support_count} depth pixels inside the live "
                        "target volume."
                    )
                live_mask_unusable = True
                print(
                    "[G1-GRASP] oracle GT AABB live mask has "
                    f"{support_count} depth pixels; reusing "
                    f"{len(cached['grasp_pose_candidates'])} cached "
                    f"GraspGenX poses into capture_id={capture_id} "
                    f"({_G1_GRASP_GROUNDING_REVISION}).",
                    flush=True,
                )
        if has_cached_poses:
            world_T_cam_cached = np.asarray(
                cached["world_T_camera"],
                dtype=np.float64,
            )
            camera_now_from_camera_cached = (
                np.linalg.inv(world_T_camera) @ world_T_cam_cached
            )
            candidates = tuple(
                _reexpress_grasp_pose_into_camera(
                    grasp_pose_from_payload(payload),
                    camera_now_from_camera_cached,
                )
                for payload in cached["grasp_pose_candidates"]
            )
            target_center_cached = np.asarray(
                cached["target_center_camera_m"],
                dtype=np.float64,
            )
            target_center_h = np.concatenate(
                [target_center_cached.reshape(3), [1.0]],
            )
            target_center_camera = (camera_now_from_camera_cached @ target_center_h)[:3]
            target_center_world = np.asarray(
                cached["target_center_world_m"],
                dtype=np.float64,
            )
            x1, y1, x2, y2 = (int(value) for value in cached["bbox_2d"])
            support_count = int(cached["support_count"])
            if not live_mask_unusable:
                print(
                    "[G1-GRASP] oracle GT AABB grounding: "
                    f"target={target_id!r} reusing {len(candidates)} cached "
                    f"GraspGenX poses into capture_id={capture_id} "
                    f"({_G1_GRASP_GROUNDING_REVISION}).",
                    flush=True,
                )
        else:
            try:
                candidates = tuple(
                    grasp_backend.propose(
                        depth_m=depth,
                        intrinsics=intrinsics,
                        target_mask=target_mask,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — no substitute grounding
                raise RuntimeError(
                    f"Oracle GT AABB GraspGenX failed ({type(exc).__name__}: {exc})."
                ) from exc
            if not candidates:
                raise RuntimeError("Oracle GT AABB GraspGenX returned no poses.")
            target_center_camera = np.median(
                points_camera[volume_supported],
                axis=0,
            )
            target_center_world = np.asarray(authored_center, dtype=np.float64)
            phase["_cached_gt_aabb_grasp"] = {
                "grasp_pose_candidates": [
                    grasp_pose_payload(candidate) for candidate in candidates
                ],
                "target_center_camera_m": [
                    float(value) for value in target_center_camera
                ],
                "target_center_world_m": [
                    float(value) for value in target_center_world
                ],
                "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                "support_count": int(support_count),
                "world_T_camera": world_T_camera.tolist(),
                "capture_id": int(capture_id),
            }
            print(
                "[G1-GRASP] oracle GT AABB grounding: "
                f"target={target_id!r} pixels={support_count} "
                f"bbox_2d={[x1, y1, x2, y2]} "
                f"authored_size_m={[round(upper[i] - lower[i], 4) for i in range(3)]} "
                f"poses={len(candidates)} "
                f"revision={_G1_GRASP_GROUNDING_REVISION}",
                flush=True,
            )
        phrase = str(phase.get("target_phrase") or target_id)
        return {
            "action": "grasp_object",
            "hand": G1_MANIPULATION_HAND,
            "preset": "grasp",
            "target": phrase,
            "thinking": (
                "Oracle GT AABB depth mask into GraspGenX, then the evaluated "
                f"right-hand grasp_object skill ({_G1_GRASP_GROUNDING_REVISION})."
            ),
            "call_name": "grasp_object",
            "envelope": self._base_envelope(),
            "controller_metadata": {
                "agent_grasp": {
                    "target": phrase,
                    "target_center_world_m": [
                        float(value) for value in target_center_world
                    ],
                    "target_center_camera_m": [
                        float(value) for value in target_center_camera
                    ],
                    "bbox_2d": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "semantic_score": 1.0,
                    "world_T_camera": world_T_camera.tolist(),
                    "capture_id": int(capture_id),
                    "grasp_pose_candidates": [
                        grasp_pose_payload(candidate) for candidate in candidates
                    ],
                    "grounding_mode": "oracle_gt_aabb",
                },
            },
            "tool_results": [
                {
                    "tool": "grasp_object",
                    "content": (
                        f"Coordinate-grounded GraspGenX produced {len(candidates)} "
                        f"poses from {support_count} GT AABB depth pixels."
                    ),
                    "is_error": False,
                    "metadata": {
                        "target": phrase,
                        "hand": G1_MANIPULATION_HAND,
                        "target_depth_pixels": support_count,
                        "grounding_mode": "oracle_gt_aabb",
                        "graspgenx_pose_count": len(candidates),
                    },
                }
            ],
        }

    def _grasp_reach(
        self,
        phase: dict[str, Any],
    ) -> dict[str, Any]:
        """Search one live target-facing hand shell before FixedJoint attach."""
        target_id = str(phase.get("target") or "")
        if not target_id:
            raise ValueError("Grasp reach phase requires a target object.")
        manipulation = self._oracle_world_snapshot.get("manipulation")
        if not isinstance(manipulation, dict):
            raise RuntimeError(
                "Grasp reach requires the private live manipulation snapshot."
            )
        objects = manipulation.get("objects")
        target_state = objects.get(target_id) if isinstance(objects, dict) else None
        if not isinstance(target_state, dict):
            raise RuntimeError(f"Grasp reach cannot resolve live target {target_id!r}.")
        if bool(target_state.get("attached", False)):
            raise RuntimeError(
                f"Grasp reach target is already attached: {target_id!r}."
            )

        pending_target = phase.get("pending_target_position_world")
        if pending_target is not None:
            last_ik = manipulation.get("last_arm_ik")
            if not isinstance(last_ik, dict):
                raise RuntimeError(
                    "Executed grasp reach produced no private IK evidence."
                )
            actual_target = last_ik.get("target_position_world")
            if (
                not isinstance(actual_target, (list, tuple))
                or len(actual_target) != 3
                or math.dist(
                    [float(value) for value in actual_target],
                    [float(value) for value in pending_target],
                )
                > 1e-6
            ):
                raise RuntimeError(
                    "Live IK evidence does not match the requested grasp "
                    f"target: requested={pending_target!r}, "
                    f"executed={actual_target!r}."
                )
            hand = str(last_ik.get("hand") or "").lower()
            if hand != G1_MANIPULATION_HAND:
                raise RuntimeError(
                    "Executed grasp reach did not use the certified right "
                    f"hand: {hand!r}."
                )
            evidence = measure_grasp_reach(
                manipulation,
                target_object_id=target_id,
                hand=hand,
                requested_target_position_world=tuple(
                    float(value) for value in pending_target
                ),
                predicted_ik_error_m=float(last_ik.get("position_error_m")),
            )
            self._grasp_reach_evidence.append(evidence.model_dump(mode="json"))
            phase.pop("pending_target_position_world", None)
            if evidence.contact_is_verified:
                self._advance_phase()
                return self._tool_decision(
                    "grasp_object",
                    {"hand": hand, "object_id": target_id},
                    (
                        "Attach the live target after native PhysX records "
                        "hand-object contact: "
                        f"grasp_center_to_target_m="
                        f"{evidence.grasp_center_to_target_m:.4f}, "
                        f"aabb_surface_gap_m="
                        f"{evidence.aabb_surface_gap_m:.4f}."
                    ),
                    hold_last_arm_ik=True,
                )
            if int(phase.get("arm_trials", 0)) >= (
                GRASP_REACH_SEARCH.maximum_arm_trials
            ):
                aabb_gaps = [
                    round(float(item["aabb_surface_gap_m"]), 4)
                    for item in self._grasp_reach_evidence
                    if item.get("target_object_id") == target_id
                ]
                raise PhysicalTrialCandidateError(
                    "G1 grasp reach exhausted its bounded physical arm "
                    "trials without native PhysX hand-object contact: "
                    f"target={target_id!r}, "
                    f"diagnostic_aabb_gaps_m={aabb_gaps!r}.",
                    stage="A",
                    reason="grasp_enclosure_not_reached",
                    evidence={
                        "target_object_id": target_id,
                        "diagnostic_aabb_gaps_m": aabb_gaps,
                    },
                )

        candidates = phase.get("hand_target_candidates")
        if candidates is None:
            lower, upper = self._live_target_bounds(
                target_state,
                target_id=target_id,
            )
            robot = self._oracle_world_snapshot.get("robot") or {}
            position = robot.get("position") if isinstance(robot, dict) else None
            if not isinstance(position, (list, tuple)) or len(position) < 2:
                raise RuntimeError("Grasp reach requires the live Robot position.")
            from tarsphere.bench.certification.grasp_reach import (
                grasp_center_target,
            )

            candidates = (
                grasp_center_target(
                    target_bounds_min_xyz=lower,
                    target_bounds_max_xyz=upper,
                ),
            )
            phase["hand_target_candidates"] = candidates
            phase["candidate_cursor"] = 0
            phase["arm_trials"] = 0

        cursor = int(phase.get("candidate_cursor", 0))
        while cursor < len(candidates):
            candidate = candidates[cursor]
            cursor += 1
            phase["candidate_cursor"] = cursor
            reachability = self._live_g1_arm_reachability(
                {
                    "target_position_world": (candidate.target_position_world),
                    "position_tolerance_m": G1_GRASP_REACH_TOLERANCE_M,
                    "align_grasp_center": True,
                }
            )
            if not bool(reachability["reachable"]):
                continue
            phase["pending_target_position_world"] = list(
                candidate.target_position_world
            )
            phase["arm_trials"] = int(phase.get("arm_trials", 0)) + 1
            return self._arm_pose_decision(
                {
                    "action": "prepare_grasp",
                    "target": target_id,
                    "hand": str(reachability["hand"]),
                    "target_position_world": candidate.target_position_world,
                    "position_tolerance_m": G1_GRASP_REACH_TOLERANCE_M,
                    "posture": "front_auto",
                    "max_facing_error_deg": G1_GRASP_ENTRY_MAX_FACING_ERROR_DEG,
                    "finish_on_target": True,
                    "finish_tolerance_rad": 0.02,
                    "align_grasp_center": True,
                    "internal_controller_action": True,
                    "pin_base_for_arm_reach": True,
                    "reason": (
                        "Execute one live target-facing right-hand grasp reach "
                        f"({phase['arm_trials']}/"
                        f"{GRASP_REACH_SEARCH.maximum_arm_trials})."
                    ),
                },
                self._current_observation,
            )

        raise PhysicalTrialCandidateError(
            "No candidate on the bounded target-facing hand shell is "
            f"reachable by the certified right G1 arm: target={target_id!r}.",
            stage="A",
            reason="no_reachable_arm_target",
            evidence={"target_object_id": target_id},
        )

    def _record_target_observation(
        self,
        phase: dict[str, Any],
    ) -> None:
        """Record the fixed task-relevant target observation opportunity."""
        target_id = str(phase.get("target") or "")
        if not target_id:
            raise ValueError("Target observation phase requires a target object.")
        manipulation = self._oracle_world_snapshot.get("manipulation")
        objects = (
            manipulation.get("objects") if isinstance(manipulation, dict) else None
        )
        target_state = objects.get(target_id) if isinstance(objects, dict) else None
        if not isinstance(target_state, dict):
            raise RuntimeError(
                "Target observation requires the private live manipulation "
                f"snapshot for {target_id!r}."
            )
        lower, upper = self._live_target_bounds(
            target_state,
            target_id=target_id,
        )
        bbox = self._live_target_visibility_bbox(
            lower,
            upper,
            target_id=target_id,
        )
        camera = self._current_observation.get("camera_params") or {}
        camera_path = str(camera.get("camera_path") or "").strip()
        if not camera_path:
            raise RuntimeError(
                "Target observation requires the calibrated ego camera path."
            )
        self._target_observation_evidence.append(
            {
                "target_object_id": target_id,
                "bbox_2d": bbox,
                "target_bounds_min_xyz": list(lower),
                "target_bounds_max_xyz": list(upper),
                "camera_path": camera_path,
            }
        )

    def _live_target_visibility_bbox(
        self,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        *,
        target_id: str,
    ) -> list[float]:
        """Require depth-supported pixels at the fixed observation opportunity."""
        bbox, depth_available = self._observed_volume_bbox(
            G.BBox3D(lower, upper),
            tolerance_m=_GOLDEN_OBJECT_DEPTH_SUPPORT_TOLERANCE_M,
        )
        if not depth_available:
            raise RuntimeError(
                "Target observation certification requires calibrated ego "
                f"RGB-D for {target_id!r}."
            )
        if bbox is None:
            raise PhysicalTrialCandidateError(
                "Live target is not depth-visible at its fixed observation "
                f"opportunity: {target_id!r}.",
                stage="A",
                reason="target_not_depth_visible",
                evidence={
                    "target_object_id": target_id,
                    **self._target_visibility_evidence(
                        G.BBox3D(lower, upper),
                    ),
                },
            )
        return bbox

    def _target_visibility_evidence(
        self,
        volume: G.BBox3D,
    ) -> dict[str, Any]:
        """Describe why one live target failed ego RGB-D visibility."""
        observation = self._current_observation
        camera = observation.get("camera_params") or {}
        transform = np.asarray(
            camera.get("world_T_cam") or [],
            dtype=np.float64,
        )
        evidence: dict[str, Any] = {
            "target_bounds_min_xyz": list(volume.min),
            "target_bounds_max_xyz": list(volume.max),
            "camera_path": camera.get("camera_path"),
            "image_size": [
                int(camera.get("width") or 0),
                int(camera.get("height") or 0),
            ],
            "geometric_projection_bbox": self._project_volume_without_depth(
                volume,
            ),
        }
        if transform.shape != (4, 4):
            return evidence

        camera_from_world = np.linalg.inv(transform)
        center_world = np.append(volume.center, 1.0)
        center_camera = camera_from_world @ center_world
        evidence.update(
            {
                "camera_world_xyz": transform[:3, 3].tolist(),
                "camera_forward_xyz": transform[:3, 2].tolist(),
                "target_center_camera_xyz": center_camera[:3].tolist(),
            }
        )

        projected = evidence["geometric_projection_bbox"]
        depth = observation.get("depth")
        if projected is None or depth is None:
            return evidence
        depth_array = np.squeeze(np.asarray(depth))
        if depth_array.ndim != 2:
            return evidence
        x1, y1, x2, y2 = projected
        left = max(0, int(math.floor(x1)))
        top = max(0, int(math.floor(y1)))
        right = min(depth_array.shape[1], int(math.ceil(x2)))
        bottom = min(depth_array.shape[0], int(math.ceil(y2)))
        crop = depth_array[top:bottom, left:right]
        valid = crop[np.isfinite(crop) & (crop > 0.0)]
        if valid.size:
            evidence["projected_depth_range_m"] = [
                float(np.min(valid)),
                float(np.max(valid)),
            ]
        return evidence

    def _authored_pickable_geometry(
        self,
        target_id: str,
    ) -> (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    ):
        """Return the frozen episode AABB used by grasp-asset certification."""

        world = getattr(self.episode, "world_knowledge", None)
        if world is None:
            world = (getattr(self, "_episode_raw", None) or {}).get("world_knowledge")
        if isinstance(world, dict):
            objects = world.get("objects")
        else:
            objects = getattr(world, "objects", None)
        for item in objects or ():
            if isinstance(item, dict):
                object_id = str(item.get("object_id") or "")
                lower_raw = item.get("bounds_min_xyz")
                upper_raw = item.get("bounds_max_xyz")
                center_raw = item.get("position_xyz")
            else:
                object_id = str(getattr(item, "object_id", "") or "")
                lower_raw = getattr(item, "bounds_min_xyz", None)
                upper_raw = getattr(item, "bounds_max_xyz", None)
                center_raw = getattr(item, "position_xyz", None)
            if object_id != target_id:
                continue
            if (
                not isinstance(lower_raw, (list, tuple))
                or not isinstance(upper_raw, (list, tuple))
                or not isinstance(center_raw, (list, tuple))
                or len(lower_raw) != 3
                or len(upper_raw) != 3
                or len(center_raw) != 3
            ):
                return None
            lower = tuple(float(value) for value in lower_raw)
            upper = tuple(float(value) for value in upper_raw)
            center = tuple(float(value) for value in center_raw)
            if not all(
                math.isfinite(value) for value in (*lower, *upper, *center)
            ) or any(lower[axis] > upper[axis] for axis in range(3)):
                return None
            return lower, upper, center
        return None

    @staticmethod
    def _live_target_bounds(
        target_state: dict[str, Any],
        *,
        target_id: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Read one finite ordered live target AABB from private evidence."""
        lower_raw = target_state.get("bbox_min")
        upper_raw = target_state.get("bbox_max")
        if (
            not isinstance(lower_raw, (list, tuple))
            or not isinstance(upper_raw, (list, tuple))
            or len(lower_raw) != 3
            or len(upper_raw) != 3
        ):
            raise RuntimeError(f"Live target bounds are missing for {target_id!r}.")
        lower = tuple(float(value) for value in lower_raw)
        upper = tuple(float(value) for value in upper_raw)
        if not all(math.isfinite(value) for value in (*lower, *upper)) or any(
            lower[axis] > upper[axis] for axis in range(3)
        ):
            raise RuntimeError(f"Live target bounds are invalid for {target_id!r}.")
        return lower, upper

    def _advance_phase(self, *, defer_gate: bool = False) -> None:
        if self.phase_index < len(self.phases):
            gate = str(self.phases[self.phase_index].get("certification_gate") or "")
            if gate:
                if defer_gate:
                    if self._pending_certification_gate is not None:
                        raise RuntimeError(
                            "Golden Oracle cannot defer two certification "
                            "gates before environment feedback."
                        )
                    self._pending_certification_gate = gate
                else:
                    self._completed_certification_gates.add(gate)
        self.phase_index += 1
        self._nav_started = False
        self._nav_path = []
        self._nav_entry_target_yaw = None

    def _communication_decision(self, phase: dict[str, Any]) -> dict[str, Any]:
        content = str(phase["content"])
        action = str(phase.get("action") or "ask")
        recipient = str(phase["recipient"])
        tool_args = {"recipient": recipient, "message": content}
        if action == "ask":
            tool_args["intent_id"] = str(phase["intent_id"])
        envelope = self._base_envelope()
        envelope["actions"] = [
            {
                "tool": action,
                "args": tool_args,
            }
        ]
        decision = {
            "action": action,
            "recipient": recipient,
            "message": content,
            "thinking": str(phase.get("reason") or "Golden oracle communication."),
            "envelope": envelope,
            "call_name": action,
        }
        if action == "ask":
            decision["intent_id"] = str(phase["intent_id"])
        return decision

    def _tool_decision(
        self,
        action: str,
        args: dict[str, Any],
        reasoning: str,
        *,
        upper_command: Any = None,
        place_ik: Any = None,
        hold_last_arm_ik: bool = False,
    ) -> dict[str, Any]:
        envelope = self._base_envelope()
        envelope_args = dict(args)
        if action in {"grasp_object", "place_object"} and args.get("object_id"):
            # Interaction permission scopes intentionally use the public
            # natural name, while the runtime tool needs the canonical id.
            record = self._world_object_record(str(args["object_id"]))
            target_name = str(
                record.get("name") or record.get("category") or ""
            ).strip()
            if not target_name:
                raise ValueError(
                    f"Physical target has no public name: {args['object_id']!r}"
                )
            envelope_args["target"] = target_name
        envelope["actions"] = [{"tool": action, "args": envelope_args}]
        decision = {
            "action": action,
            **args,
            "thinking": reasoning,
            "envelope": envelope,
            "call_name": action,
        }
        if upper_command is not None:
            command = [float(value) for value in upper_command]
            decision["controller_action"] = {
                "move_by_speed": [0.0, 0.0, 0.0, 0.74, command],
            }
            decision["controller_backend"] = "move_by_speed"
            decision["controller_metadata"] = {
                "skill_action": {"action": action, **args},
                "upper_body_command": command,
            }
        elif isinstance(place_ik, dict):
            decision["controller_metadata"] = {
                "skill_action": {"action": action, **args},
                "g1_place_ik": dict(place_ik),
            }
        elif hold_last_arm_ik:
            decision["controller_metadata"] = {
                "skill_action": {"action": action, **args},
                "g1_arm_ik_hold": True,
            }
        return decision

    def _base_envelope(self) -> dict[str, Any]:
        """Emit evaluator inputs required by the current public task event."""
        envelope: dict[str, Any] = {
            "thinking": "Benchmark-private Golden authored policy certification.",
        }
        requirements = self._decision_requirements
        if requirements is None:
            return envelope

        if isinstance(getattr(requirements, "planning", None), dict):
            evaluation = self.episode.evaluation
            if evaluation is None:
                raise RuntimeError("Golden planning output requires episode.evaluation")
            planning_milestones = tuple(
                milestone
                for milestone in evaluation.milestones
                if milestone.outcome.evaluator == "structured_plan_submitted"
            )
            if len(planning_milestones) != 1:
                raise RuntimeError(
                    "Golden episode must define exactly one initial-plan "
                    f"milestone; found {len(planning_milestones)}"
                )
            ground_truth = planning_milestones[0].outcome.ground_truth
            envelope["planning"] = {
                "goal": str(ground_truth.get("goal") or "").strip(),
                "steps": list(ground_truth.get("required_subgoals") or ()),
            }
        return envelope

    def _world_object_record(self, object_id: str) -> dict[str, Any]:
        world = self.episode.world_knowledge or {}
        candidates = world.get("objects") or []
        return next(
            (
                dict(item)
                for item in candidates
                if str(item.get("object_id") or "") == object_id
            ),
            {},
        )

    def _human_runtime_state(self, character_id: str) -> dict[str, Any]:
        humans = self._oracle_world_snapshot.get("humans") or {}
        live = humans.get(character_id) if isinstance(humans, dict) else None
        if isinstance(live, dict) and live.get("position") is not None:
            return dict(live)
        for human in self._human_specs:
            human_id = str(getattr(human, "id", ""))
            if human_id == character_id:
                return {
                    "position": list(getattr(human, "start_position", []) or []),
                    "animation_state": "standing",
                }
        return {}

    def _semantic_tight_bbox(self, semantic_label: str) -> list[float] | None:
        """Return the union of tight boxes whose semantic label matches exactly."""

        annotation = self._current_observation.get("bounding_box_2d_tight")
        if not isinstance(annotation, dict):
            return None
        info = annotation.get("info")
        labels_by_id = info.get("idToLabels") if isinstance(info, dict) else None
        if not isinstance(labels_by_id, dict):
            return None

        wanted = str(semantic_label).strip()
        matching_ids: set[int] = set()
        for raw_id, raw_labels in labels_by_id.items():
            if wanted not in self._semantic_label_values(raw_labels):
                continue
            try:
                matching_ids.add(int(raw_id))
            except (TypeError, ValueError):
                continue
        if not matching_ids:
            return None

        rows = annotation.get("data")
        if rows is None:
            return None
        boxes = [
            parsed[1:]
            for row in rows
            if (parsed := self._tight_bbox_row(row)) is not None
            and parsed[0] in matching_ids
        ]
        if not boxes:
            return None

        camera = self._current_observation.get("camera_params") or {}
        width = int(camera.get("width") or 0)
        height = int(camera.get("height") or 0)
        if width <= 0 or height <= 0:
            return None
        x1 = max(0.0, min(box[0] for box in boxes))
        y1 = max(0.0, min(box[1] for box in boxes))
        x2 = min(float(width - 1), max(box[2] for box in boxes))
        y2 = min(float(height - 1), max(box[3] for box in boxes))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _semantic_label_values(raw_labels: Any) -> set[str]:
        if isinstance(raw_labels, dict):
            values = raw_labels.values()
        elif isinstance(raw_labels, (list, tuple, set)):
            values = raw_labels
        else:
            value = str(raw_labels).strip()
            return {value} if value else set()
        return {str(value).strip() for value in values if str(value).strip()}

    @staticmethod
    def _tight_bbox_row(
        row: Any,
    ) -> tuple[int, float, float, float, float] | None:
        names = getattr(getattr(row, "dtype", None), "names", None)
        try:
            if names:
                fields = {str(name).casefold(): name for name in names}

                def field(*candidates: str) -> Any:
                    name = next(
                        (fields[key] for key in candidates if key in fields),
                        None,
                    )
                    if name is None:
                        raise KeyError(candidates[0])
                    return row[name]

                return (
                    int(field("semanticid", "semantic_id", "id")),
                    float(field("x_min", "xmin", "x1")),
                    float(field("y_min", "ymin", "y1")),
                    float(field("x_max", "xmax", "x2")),
                    float(field("y_max", "ymax", "y2")),
                )
            values = list(row)
            if len(values) < 5:
                return None
            return (
                int(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def _project_world_bbox(self, record: dict[str, Any]) -> list[float] | None:
        """Project a world AABB only when it contributes observable pixels."""

        lower = record.get("bounds_min_xyz")
        upper = record.get("bounds_max_xyz")
        if not isinstance(lower, (list, tuple)) or not isinstance(upper, (list, tuple)):
            return None
        if len(lower) < 3 or len(upper) < 3:
            return None
        volume = G.BBox3D(tuple(lower[:3]), tuple(upper[:3]))
        observed, depth_available = self._observed_volume_bbox(
            volume,
            tolerance_m=_GOLDEN_OBJECT_DEPTH_SUPPORT_TOLERANCE_M,
        )
        if observed is not None:
            return observed
        if depth_available:
            return None
        camera = self._current_observation.get("camera_params") or {}
        width = int(camera.get("width") or 0)
        height = int(camera.get("height") or 0)
        return self._project_volume(
            volume,
            min_projected_area_px=(
                _GOLDEN_OBJECT_MIN_PROJECTED_AREA_RATIO * width * height
            ),
        )

    def _observed_volume_bbox(
        self,
        volume: G.BBox3D,
        *,
        tolerance_m: float = 0.05,
    ) -> tuple[list[float] | None, bool]:
        """Return a depth-supported visible box and whether depth was usable."""
        observation = self._current_observation
        depth = observation.get("depth")
        camera = observation.get("camera_params") or {}
        transform_raw = camera.get("world_T_cam")
        width = int(camera.get("width") or 0)
        height = int(camera.get("height") or 0)
        if depth is None or transform_raw is None or width <= 0 or height <= 0:
            return None, False
        depth_array = np.asarray(depth)
        transform = np.asarray(transform_raw, dtype=np.float64)
        if depth_array.size == 0 or transform.shape != (4, 4):
            return None, False
        intrinsics = G.fov_to_intrinsics(
            float(camera.get("fov_deg") or 90.0),
            width,
            height,
        )
        bbox = G.observed_volume_bbox(
            volume,
            depth_array,
            intrinsics,
            transform,
            tolerance_m=float(tolerance_m),
        )
        return (list(bbox) if bbox is not None else None), True

    def _project_volume(
        self,
        volume: G.BBox3D,
        *,
        min_projected_area_px: float = 0.0,
    ) -> list[float] | None:
        observation = self._current_observation
        pose = observation.get("pose") or {}
        camera = observation.get("camera_params") or {}
        position = pose.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return None
        width = int(camera.get("width") or 0)
        height = int(camera.get("height") or 0)
        fov = float(camera.get("fov_deg") or 90.0)
        if width <= 0 or height <= 0:
            return None
        transform_raw = camera.get("world_T_cam")
        if transform_raw is None:
            return None
        transform = np.asarray(transform_raw, dtype=np.float64)
        if transform.shape != (4, 4):
            return None
        intrinsics = G.fov_to_intrinsics(fov, width, height)
        depth = observation.get("depth")
        if not G.is_visible_in_frustum(
            volume,
            transform,
            intrinsics,
            (height, width),
            depth=depth,
        ):
            return None
        return self._project_volume_without_depth(
            volume,
            min_projected_area_px=min_projected_area_px,
        )

    def _project_volume_without_depth(
        self,
        volume: G.BBox3D,
        *,
        min_projected_area_px: float = 0.0,
    ) -> list[float] | None:
        """Project one volume using camera geometry without occlusion claims."""
        observation = self._current_observation
        camera = observation.get("camera_params") or {}
        width = int(camera.get("width") or 0)
        height = int(camera.get("height") or 0)
        fov = float(camera.get("fov_deg") or 90.0)
        if width <= 0 or height <= 0:
            return None
        transform_raw = camera.get("world_T_cam")
        if transform_raw is None:
            return None
        transform = np.asarray(transform_raw, dtype=np.float64)
        if transform.shape != (4, 4):
            return None
        intrinsics = G.fov_to_intrinsics(fov, width, height)
        if not G.is_visible_in_frustum(
            volume,
            transform,
            intrinsics,
            (height, width),
        ):
            return None
        lower = np.asarray(volume.min, dtype=float)
        upper = np.asarray(volume.max, dtype=float)
        corners = np.asarray(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=float,
        )
        camera_from_world = np.linalg.inv(transform)
        homogeneous = np.concatenate(
            [corners, np.ones((len(corners), 1), dtype=float)], axis=1
        )
        camera_points = (camera_from_world @ homogeneous.T).T[:, :3]
        camera_points = camera_points[camera_points[:, 2] > 0.1]
        if not len(camera_points):
            return None
        projected = camera_points @ intrinsics.T
        pixels = projected[:, :2] / camera_points[:, 2:3]
        x1 = max(0.0, float(np.min(pixels[:, 0])))
        y1 = max(0.0, float(np.min(pixels[:, 1])))
        x2 = min(float(width - 1), float(np.max(pixels[:, 0])))
        y2 = min(float(height - 1), float(np.max(pixels[:, 1])))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            return None
        if (x2 - x1) * (y2 - y1) < float(min_projected_area_px):
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def build_grasp_approach_phases(episode: Any) -> list[dict[str, Any]]:
        """Compile oracle locomotion to ``physical_task.grasp`` stance.

        Reuses the same ``_nav_phase`` / ``_precise_phase`` / ``_orient_phase``
        helpers as physical-task certification. Missing ``physical_task.grasp``
        fails immediately.
        """
        world = getattr(episode, "world_knowledge", None) or {}
        if not isinstance(world, dict):
            raise RuntimeError(
                "oracle_approach requires episode.world_knowledge to be a mapping."
            )
        task = world.get("physical_task")
        if not isinstance(task, dict) or not task:
            raise RuntimeError(
                "oracle_approach requires world_knowledge.physical_task.grasp."
            )
        grasp = task.get("grasp")
        if not isinstance(grasp, dict) or not grasp:
            raise RuntimeError(
                "oracle_approach requires world_knowledge.physical_task.grasp. "
                "Missing grasp stance; will not invent a walk."
            )
        stance = grasp.get("robot_stance_xy")
        if not isinstance(stance, (list, tuple)) or len(stance) < 2:
            raise RuntimeError(
                "oracle_approach requires physical_task.grasp.robot_stance_xy."
            )
        objects = world.get("objects")
        if (
            not isinstance(objects, Sequence)
            or isinstance(objects, (str, bytes))
            or not objects
        ):
            raise RuntimeError(
                "oracle_approach requires world_knowledge.objects so the "
                "oracle can face the grasp target."
            )
        objects_by_id = {
            str(item["object_id"]): dict(item)
            for item in objects
            if isinstance(item, dict) and item.get("object_id")
        }
        target_id = str(
            grasp.get("target_object_id") or task.get("target_object_id") or ""
        )
        if not target_id or target_id not in objects_by_id:
            raise RuntimeError(
                "oracle_approach cannot resolve physical_task.grasp."
                f"target_object_id {target_id!r} in world_knowledge.objects."
            )
        target = objects_by_id[target_id]
        phases: list[dict[str, Any]] = []
        route = world.get("episode_route")
        nav = route.get("navigate_to_target") if isinstance(route, dict) else None
        points = nav.get("points") if isinstance(nav, dict) else None
        if (
            isinstance(points, Sequence)
            and not isinstance(points, (str, bytes))
            and len(points) >= 2
        ):
            for point in points[1:]:
                if not isinstance(point, dict):
                    raise RuntimeError(
                        "oracle_approach requires authored navigate_to_target "
                        f"points to be mappings; got {point!r}."
                    )
                point_id = str(point.get("point_id") or "grasp navigation")
                phases.append(
                    GoldenOracleAgent._nav_phase(
                        point["position_xy"],
                        point_id.replace("_", " "),
                        tolerance_m=NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
                    )
                )
        if not phases:
            approach = grasp.get("navigation_approach_xy") or stance
            phases.append(
                GoldenOracleAgent._nav_phase(
                    approach,
                    "grasp navigation approach",
                    tolerance_m=NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
                )
            )
        phases.append(
            GoldenOracleAgent._precise_phase(
                stance,
                "collision-safe target grasp stance",
                tolerance_m=_G1_MANIPULATION_STANCE_TOLERANCE_M,
            )
        )
        phases.append(
            GoldenOracleAgent._orient_phase(
                target,
                stance=stance,
                label="target grasp alignment",
                stance_tolerance_m=_G1_MANIPULATION_STANCE_TOLERANCE_M,
            )
        )
        return phases

    @staticmethod
    def build_oracle_grasp_phases(episode: Any) -> list[dict[str, Any]]:
        """Compile approach phases plus the public GT AABB ``grasp_reach``.

        Certification's public grasp path is ``precise_approach`` →
        ``orient`` → ``tilt_camera`` (inside ``_public_grasp_object``) →
        ``grasp_object`` with ``grounding_mode='oracle_gt_aabb'``. Crouch
        is compiled only for the private-IK path.
        """
        phases = GoldenOracleAgent.build_grasp_approach_phases(episode)
        world = getattr(episode, "world_knowledge", None) or {}
        task = world.get("physical_task") if isinstance(world, dict) else None
        if not isinstance(task, dict):
            raise RuntimeError(
                "oracle_grasp requires world_knowledge.physical_task.grasp."
            )
        grasp = task.get("grasp")
        if not isinstance(grasp, dict) or not grasp:
            raise RuntimeError(
                "oracle_grasp requires world_knowledge.physical_task.grasp."
            )
        objects = world.get("objects")
        objects_by_id = {
            str(item["object_id"]): dict(item)
            for item in objects or ()
            if isinstance(item, dict) and item.get("object_id")
        }
        target_id = str(
            grasp.get("target_object_id") or task.get("target_object_id") or ""
        )
        if not target_id or target_id not in objects_by_id:
            raise RuntimeError(
                "oracle_grasp cannot resolve physical_task.grasp."
                f"target_object_id {target_id!r} in world_knowledge.objects."
            )
        compiled = GoldenOracleAgent._compiled_grasp_phases(
            grasp,
            target_id=target_id,
            target=objects_by_id[target_id],
            public_skill=True,
        )
        reach = [
            phase for phase in compiled if str(phase.get("kind") or "") == "grasp_reach"
        ]
        if len(reach) != 1 or not reach[0].get("public_skill"):
            raise RuntimeError(
                "oracle_grasp requires exactly one public grasp_reach phase "
                "from GoldenOracleAgent._compiled_grasp_phases."
            )
        phases.extend(reach)
        return phases

    @staticmethod
    def _build_physical_task_phases(episode: Any) -> list[dict[str, Any]]:
        """Compile the episode's frozen physical task without capability detours."""
        world = episode.world_knowledge or {}
        task = world.get("physical_task")
        route = world.get("episode_route")
        if not isinstance(task, dict):
            raise ValueError(
                "Physical-task-only certification requires "
                "world_knowledge.physical_task."
            )
        if not isinstance(route, dict):
            raise ValueError(
                "Physical-task-only certification requires "
                "world_knowledge.episode_route."
            )
        return GoldenOracleAgent._build_physical_trial_phases(
            episode,
            {
                "grasp": task.get("grasp"),
                "place": task.get("place"),
                "navigate_to_target": route.get("navigate_to_target"),
                "transport_to_destination": route.get("transport_to_destination"),
            },
        )

    @staticmethod
    def _build_phases(episode: Any) -> list[dict[str, Any]]:
        world = episode.world_knowledge or {}
        grasp_motion_trial = world.get("grasp_motion_trial")
        grasp_system_trial = world.get("grasp_system_trial")
        physical_trial = world.get("physical_trial")
        if grasp_motion_trial is not None:
            if any(
                world.get(key) is not None
                for key in (
                    "grasp_system_trial",
                    "physical_trial",
                    "physical_task",
                    "episode_route",
                )
            ):
                raise ValueError(
                    "Grasp-motion trials cannot declare another task or route"
                )
            return GoldenOracleAgent._build_grasp_motion_phases(
                episode,
                grasp_motion_trial,
            )
        if grasp_system_trial is not None:
            if any(
                world.get(key) is not None
                for key in (
                    "grasp_motion_trial",
                    "physical_trial",
                    "physical_task",
                    "episode_route",
                )
            ):
                raise ValueError(
                    "Grasp-system trials cannot declare another task or route"
                )
            if not isinstance(grasp_system_trial, dict):
                raise ValueError("world_knowledge.grasp_system_trial must be a mapping")
            reference = grasp_system_trial.get("reference_object")
            if not isinstance(reference, dict):
                raise ValueError("Grasp-system trial requires its reference object")
            return GoldenOracleAgent._build_grasp_system_phases(
                episode,
                grasp_system_trial,
            )
        if physical_trial is not None:
            if (
                world.get("physical_task") is not None
                or world.get("episode_route") is not None
            ):
                raise ValueError(
                    "Physical-trial episodes cannot also declare a compiled "
                    "semantic physical task or episode route"
                )
            return GoldenOracleAgent._build_physical_trial_phases(
                episode,
                physical_trial,
            )

        task = world.get("physical_task")
        episode_route = world.get("episode_route")
        objects = world.get("objects")
        if not isinstance(task, dict):
            raise ValueError("Golden Oracle requires world_knowledge.physical_task")
        if not isinstance(episode_route, dict):
            raise ValueError("Golden Oracle requires world_knowledge.episode_route")
        if (
            not isinstance(objects, Sequence)
            or isinstance(objects, (str, bytes))
            or not objects
        ):
            raise ValueError("Golden Oracle requires world_knowledge.objects")

        objects_by_id = {str(item["object_id"]): dict(item) for item in objects}
        target_id = str(task["target_object_id"])
        target = objects_by_id[target_id]
        route_points = [
            *(episode_route["navigate_to_target"]["points"]),
            *(episode_route["transport_to_destination"]["points"]),
        ]
        route_indices = {
            str(point["point_id"]): index for index, point in enumerate(route_points)
        }
        milestone_route_points = {
            str(stop["milestone_id"]): str(stop["route_point_id"])
            for stop in episode_route["milestone_stops"]
        }
        humans_by_id = {str(human.id): human for human in episode.humans}
        current_human_positions = {
            human_id: list(human.start_position)
            for human_id, human in humans_by_id.items()
        }
        phases: list[dict[str, Any]] = []
        current_route_index = 0
        grasp_compiled = False
        transport_gate_assigned = False
        evaluation = getattr(episode, "evaluation", None)
        semantic_milestones = tuple(getattr(evaluation, "milestones", None) or ())
        if not semantic_milestones:
            raise ValueError("Golden episode requires evaluation.milestones")
        object_observation_milestones = {
            str(milestone.milestone_id)
            for milestone in semantic_milestones
            if str(milestone.milestone_id) == "target_observed"
        }
        if len(object_observation_milestones) != 1:
            raise ValueError(
                "Golden episode must contain exactly one target_observation milestone."
            )

        task_requesters = [
            human
            for human in humans_by_id.values()
            if str(getattr(human, "role", "") or "") == "task_requester"
        ]
        if len(task_requesters) != 1:
            raise ValueError(
                "Golden episode must contain exactly one task_requester Human"
            )
        social_human = task_requesters[0]
        phases.append(
            {
                "kind": "communicate",
                "recipient": str(social_human.name),
                "action": "speak",
                "content": (
                    "I understand the request. I will resolve the missing target "
                    "and permission information before acting."
                ),
                "reason": "Acknowledge the request at the authored encounter.",
            }
        )

        for milestone_index, milestone in enumerate(semantic_milestones):
            if milestone_index == 0:
                continue
            milestone_id = str(milestone.milestone_id)
            conditions = list(milestone.preconditions)
            route_point_id = milestone_route_points.get(milestone_id)
            if route_point_id is None:
                spatial_positions = [
                    condition.arguments.get("position")
                    for condition in conditions
                    if isinstance(condition, MilestoneCondition)
                    and condition.detector == "robot_in_place"
                    and condition.arguments.get("position") is not None
                ]
                matching_points = [
                    str(point["point_id"])
                    for point in route_points
                    if route_indices[str(point["point_id"])] >= current_route_index
                    and any(
                        math.dist(point["position_xy"], position[:2])
                        <= NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M
                        for position in spatial_positions
                    )
                ]
                route_point_id = matching_points[0] if matching_points else None
            destination_index = (
                route_indices[route_point_id]
                if route_point_id is not None
                else current_route_index
            )
            if destination_index < current_route_index:
                raise ValueError(
                    f"Milestone {milestone_id!r} moves backward along the "
                    "compiled episode route"
                )
            for route_index in range(
                current_route_index + 1,
                destination_index + 1,
            ):
                point = route_points[route_index]
                previous = route_points[route_index - 1]
                if (
                    math.dist(
                        point["position_xy"],
                        previous["position_xy"],
                    )
                    <= 1e-9
                ):
                    continue
                phase = GoldenOracleAgent._nav_phase(
                    point["position_xy"],
                    str(point["point_id"]).replace("_", " "),
                    tolerance_m=(
                        NAVIGATION_CONTROLLER_REACH_TOLERANCE_M
                        if (
                            route_index == destination_index
                            and milestone_id in object_observation_milestones
                        )
                        else NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M
                    ),
                )
                if grasp_compiled and not transport_gate_assigned:
                    phase["certification_gate"] = "B"
                    transport_gate_assigned = True
                phases.append(phase)
            current_route_index = destination_index

            if milestone_id in object_observation_milestones:
                robot_position = route_points[current_route_index]["position_xy"]
                phases.extend(
                    [
                        GoldenOracleAgent._orient_phase(
                            target,
                            stance=robot_position,
                            label="target observation alignment",
                            stance_tolerance_m=(
                                NAVIGATION_CONTROLLER_REACH_TOLERANCE_M
                            ),
                        ),
                        {
                            "kind": "target_observation",
                            "target": target_id,
                        },
                    ]
                )

            question_bindings = [
                dict(condition.arguments)
                for condition in conditions
                if isinstance(condition, MilestoneCondition)
                and condition.detector == "agent_interaction_selected"
            ]
            if milestone.outcome.evaluator == "agent_interaction_selected":
                question_bindings.append(dict(milestone.outcome.ground_truth))
            if question_bindings:
                if len(question_bindings) != 1:
                    raise ValueError(
                        f"Milestone {milestone_id!r} has multiple Agent "
                        "question triggers"
                    )
                question = question_bindings[0]
                human_id = str(question.get("human_id") or "")
                human = humans_by_id.get(human_id)
                if human is None:
                    raise ValueError(
                        f"Milestone {milestone_id!r} question "
                        f"references unknown Human {human_id!r}"
                    )
                robot_position = route_points[current_route_index]["position_xy"]
                human_position = current_human_positions[human_id]
                phases.append(
                    GoldenOracleAgent._orient_phase(
                        target,
                        stance=robot_position,
                        target_xy=human_position,
                        label=f"{human.name} communication encounter",
                        semantic_target=str(human.name),
                        stance_tolerance_m=(NAVIGATION_CONTROLLER_REACH_TOLERANCE_M),
                    )
                )
                intent_id = str(question.get("intent_id") or "")
                intent = next(
                    (
                        item
                        for item in episode.interaction.intents
                        if item.intent_id == intent_id
                    ),
                    None,
                )
                if intent is None:
                    raise ValueError(
                        f"Milestone {milestone_id!r} references undeclared "
                        f"interaction intent {intent_id!r}"
                    )
                phases.append(
                    {
                        "kind": "communicate",
                        "intent_id": intent_id,
                        "recipient": str(human.name),
                        "content": f"{human.name}, {intent.description}",
                        "reason": (
                            f"Ask the responsible Human for {intent_id} before "
                            "continuing the physical task."
                        ),
                    }
                )
                phases.append(
                    {
                        "kind": "wait",
                        "remaining": 1,
                        "reason": (f"Observe {human.name} response before continuing."),
                    }
                )
            else:
                phases.append(
                    {
                        "kind": "wait",
                        "remaining": 1,
                        "reason": (
                            f"Expose milestone {milestone_id} before the next "
                            "task transition."
                        ),
                    }
                )

            for human_action in milestone.event.human_actions:
                if human_action.get("command") != "goto":
                    continue
                human_id = str(human_action.get("human") or "")
                anchor_id = str(human_action.get("target") or "")
                anchor = (world.get("anchors") or {}).get(anchor_id)
                center = anchor.get("center") if isinstance(anchor, dict) else None
                if human_id in current_human_positions and center is not None:
                    current_human_positions[human_id] = list(center)

            if milestone_id == "target_observed":
                if grasp_compiled:
                    raise ValueError(
                        "Golden episode contains multiple physical execution "
                        "entry milestones"
                    )
                grasp_approach = task["grasp"].get("navigation_approach_xy")
                approach_candidates = [
                    index
                    for index, point in enumerate(route_points)
                    if index >= current_route_index
                    and grasp_approach is not None
                    and math.dist(point["position_xy"], grasp_approach[:2])
                    <= NAVIGATION_CONTROLLER_REACH_TOLERANCE_M
                ]
                if not approach_candidates:
                    raise ValueError(
                        "Golden grasp has no reachable navigation approach "
                        "on the compiled episode route"
                    )
                grasp_approach_index = approach_candidates[0]
                for route_index in range(
                    current_route_index + 1,
                    grasp_approach_index + 1,
                ):
                    point = route_points[route_index]
                    previous = route_points[route_index - 1]
                    if math.dist(point["position_xy"], previous["position_xy"]) <= 1e-9:
                        continue
                    phases.append(
                        GoldenOracleAgent._nav_phase(
                            point["position_xy"],
                            str(point["point_id"]).replace("_", " "),
                        )
                    )
                current_route_index = grasp_approach_index
                phases.extend(
                    GoldenOracleAgent._compiled_grasp_phases(
                        task["grasp"],
                        target_id=target_id,
                        target=target,
                    )
                )
                grasp_compiled = True

        if not grasp_compiled:
            raise ValueError("Golden episode has no physical execution entry milestone")
        for route_index in range(
            current_route_index + 1,
            len(route_points),
        ):
            point = route_points[route_index]
            previous = route_points[route_index - 1]
            if (
                math.dist(
                    point["position_xy"],
                    previous["position_xy"],
                )
                <= 1e-9
            ):
                continue
            phase = GoldenOracleAgent._nav_phase(
                point["position_xy"],
                str(point["point_id"]).replace("_", " "),
            )
            if not transport_gate_assigned:
                phase["certification_gate"] = "B"
                transport_gate_assigned = True
            phases.append(phase)

        phases.extend(
            GoldenOracleAgent._compiled_place_phases(
                task["place"],
                target_id=target_id,
                target=target,
            )
        )
        return phases

    @staticmethod
    def _build_physical_trial_phases(
        episode: Any,
        trial: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(trial, dict):
            raise ValueError(
                "Golden Oracle requires world_knowledge.physical_trial to be a mapping"
            )
        objects = (episode.world_knowledge or {}).get("objects")
        if (
            not isinstance(objects, Sequence)
            or isinstance(objects, (str, bytes))
            or not objects
        ):
            raise ValueError("Physical trial requires world_knowledge.objects")
        objects_by_id = {str(item["object_id"]): dict(item) for item in objects}
        grasp = trial.get("grasp")
        place = trial.get("place")
        navigate_to_target = trial.get("navigate_to_target")
        transport_to_destination = trial.get("transport_to_destination")
        if not all(
            isinstance(value, dict)
            for value in (
                grasp,
                place,
                navigate_to_target,
                transport_to_destination,
            )
        ):
            raise ValueError(
                "Physical trial requires grasp, place, and both navigation routes"
            )
        target_id = str(grasp.get("target_object_id") or "")
        if not target_id or target_id not in objects_by_id:
            raise ValueError(
                "Physical trial grasp target does not resolve to a runtime object"
            )
        target = objects_by_id[target_id]
        handover = episode_success_requires_handover(episode)
        public_grasp = episode_success_requires_retained_object(episode) or handover
        phases = []
        target_points = navigate_to_target.get("points")
        transport_points = transport_to_destination.get("points")
        if (
            not isinstance(target_points, Sequence)
            or isinstance(target_points, (str, bytes))
            or len(target_points) < 2
            or not isinstance(transport_points, Sequence)
            or isinstance(transport_points, (str, bytes))
            or len(transport_points) < 2
        ):
            raise ValueError(
                "Physical trial navigation routes require at least two points"
            )
        for point in target_points[1:]:
            point_id = str(point.get("point_id") or "")
            phases.append(
                GoldenOracleAgent._nav_phase(
                    point["position_xy"],
                    point_id.replace("_", " "),
                    tolerance_m=NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
                )
            )
        phases.extend(
            GoldenOracleAgent._compiled_grasp_phases(
                grasp,
                target_id=target_id,
                target=target,
                public_skill=public_grasp,
            )
        )
        transport_phases = [
            GoldenOracleAgent._nav_phase(
                point["position_xy"],
                str(point["point_id"]).replace("_", " "),
            )
            for point in transport_points[1:]
        ]
        if not transport_phases:
            raise ValueError(
                "Physical trial must contain a non-empty loaded transport leg"
            )
        if handover:
            phases.extend(transport_phases)
            handover_phases = GoldenOracleAgent._compiled_handover_phases(
                episode,
                approach_from_xy=transport_phases[-1]["goal"],
            )
            # Stage B must include the claim. Leaving B on the last transport
            # waypoint would truncate certification before the transfer.
            handover_phases[-1]["certification_gate"] = "B"
            phases.extend(handover_phases)
        else:
            transport_phases[-1]["certification_gate"] = "B"
            phases.extend(transport_phases)
            if not public_grasp:
                phases.extend(
                    GoldenOracleAgent._compiled_place_phases(
                        place,
                        target_id=target_id,
                        target=target,
                    )
                )
        return phases

    @staticmethod
    def _build_grasp_motion_phases(
        episode: Any,
        trial: Any,
    ) -> list[dict[str, Any]]:
        """Compile short navigation, real grasp, retraction, and return."""
        if not isinstance(trial, dict):
            raise ValueError(
                "Golden Oracle requires world_knowledge.grasp_motion_trial "
                "to be a mapping"
            )
        objects = (episode.world_knowledge or {}).get("objects")
        if (
            not isinstance(objects, Sequence)
            or isinstance(objects, (str, bytes))
            or not objects
        ):
            raise ValueError("Grasp-motion trial requires world_knowledge.objects")
        objects_by_id = {str(item["object_id"]): dict(item) for item in objects}
        grasp = trial.get("grasp")
        navigate_to_target = trial.get("navigate_to_target")
        return_from_target = trial.get("return_from_target")
        if not all(
            isinstance(value, dict)
            for value in (
                grasp,
                navigate_to_target,
                return_from_target,
            )
        ):
            raise ValueError(
                "Grasp-motion trial requires grasp and both navigation routes"
            )
        target_id = str(trial.get("target_object_id") or "")
        if (
            not target_id
            or target_id != str(grasp.get("target_object_id") or "")
            or target_id not in objects_by_id
        ):
            raise ValueError(
                "Grasp-motion target does not resolve to one runtime object"
            )
        approach_points = navigate_to_target.get("points")
        return_points = return_from_target.get("points")
        if any(
            not isinstance(points, Sequence)
            or isinstance(points, (str, bytes))
            or len(points) < 2
            for points in (approach_points, return_points)
        ):
            raise ValueError(
                "Grasp-motion navigation routes require at least two points"
            )

        phases = [
            GoldenOracleAgent._nav_phase(
                point["position_xy"],
                str(point["point_id"]).replace("_", " "),
                tolerance_m=NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
            )
            for point in approach_points[1:]
        ]
        phases.extend(
            GoldenOracleAgent._compiled_grasp_phases(
                grasp,
                target_id=target_id,
                target=objects_by_id[target_id],
            )
        )
        return_phases = [
            GoldenOracleAgent._nav_phase(
                point["position_xy"],
                str(point["point_id"]).replace("_", " "),
            )
            for point in return_points[1:]
        ]
        return_phases[-1]["certification_gate"] = "B"
        phases.extend(return_phases)
        phases.append({"kind": "report"})
        return phases

    def _build_grasp_system_phases(
        episode: Any,
        trial: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Compile the fixed near-field reference through the grasp stack."""
        reference = trial.get("reference_object")
        grasp = trial.get("grasp")
        if not all(isinstance(value, dict) for value in (reference, grasp)):
            raise ValueError("Grasp-system trial requires reference and grasp geometry")
        objects = (episode.world_knowledge or {}).get("objects")
        objects_by_id = {str(item["object_id"]): dict(item) for item in (objects or [])}
        target_id = str(reference.get("object_id") or "")
        if target_id not in objects_by_id:
            raise ValueError(
                "Grasp-system reference does not resolve to a runtime object"
            )
        phases = GoldenOracleAgent._compiled_grasp_phases(
            grasp,
            target_id=target_id,
            target=objects_by_id[target_id],
        )
        phases.append({"kind": "report"})
        return phases

    @staticmethod
    def _handover_success_ids(episode: Any) -> tuple[str, str]:
        success = getattr(episode, "diagnostics", None)
        matches = [
            check
            for check in getattr(success, "checks", None) or ()
            if str(getattr(check, "type", "") or "") == ("handover_accepted")
        ]
        if len(matches) != 1:
            raise ValueError(
                "Handover certification requires exactly one "
                "handover_accepted diagnostic check"
            )
        params = getattr(matches[0], "params", None) or {}
        human_id = str(params.get("human_id") or "").strip()
        object_id = str(params.get("object_id") or "").strip()
        if not human_id or not object_id:
            raise ValueError(
                "handover_accepted requires params.human_id and params.object_id"
            )
        return human_id, object_id

    @staticmethod
    def _handover_recipient(episode: Any) -> Any:
        human_id, _ = GoldenOracleAgent._handover_success_ids(episode)
        for human in getattr(episode, "humans", None) or ():
            if str(getattr(human, "id", "") or "") == human_id:
                return human
        raise ValueError(f"Handover recipient {human_id!r} is not an authored Human")

    @staticmethod
    def _authored_human_xy(human: Any) -> tuple[float, float]:
        start = getattr(human, "start_position", None) or ()
        if len(start) < 2:
            raise ValueError("Handover recipient has no authored start_position XY")
        position = (float(start[0]), float(start[1]))
        if not all(math.isfinite(value) for value in position):
            raise ValueError("Handover recipient start_position must be finite XY")
        return position

    @staticmethod
    def _preferred_handover_standoff(
        human_xy: Sequence[float],
        approach_from_xy: Sequence[float],
    ) -> tuple[float, float]:
        origin = (float(human_xy[0]), float(human_xy[1]))
        incoming = (float(approach_from_xy[0]), float(approach_from_xy[1]))
        delta = (incoming[0] - origin[0], incoming[1] - origin[1])
        length = math.hypot(*delta)
        if length <= 1e-9:
            delta, length = (0.0, 1.0), 1.0
        return (
            origin[0] + _HANDOVER_STANDOFF_M * delta[0] / length,
            origin[1] + _HANDOVER_STANDOFF_M * delta[1] / length,
        )

    @staticmethod
    def _handover_standoff_is_usable(
        planner: Planner,
        *,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        human_xy: tuple[float, float],
    ) -> bool:
        distance = math.dist(goal_xy, human_xy)
        if (
            distance - _HANDOVER_STANDOFF_TOLERANCE_M < HANDOVER_MIN_DISTANCE_M
            or distance + _HANDOVER_STANDOFF_TOLERANCE_M > HANDOVER_MAX_DISTANCE_M
        ):
            return False
        try:
            require_exact_occupancy_point(
                planner,
                goal_xy,
                label="handover standoff",
            )
        except ValueError:
            return False
        result = planner.plan(
            PlanInput(
                start_xy=start_xy,
                start_yaw_rad=0.0,
                goal_xy=goal_xy,
            )
        )
        return route_reaches_exact_goal(
            result,
            start_xy=start_xy,
            goal_xy=goal_xy,
        )

    @staticmethod
    def _reachable_handover_standoff(
        episode: Any,
        *,
        human_xy: tuple[float, float],
        approach_from_xy: Sequence[float],
    ) -> tuple[float, float]:
        start_xy = (
            float(approach_from_xy[0]),
            float(approach_from_xy[1]),
        )
        preferred = GoldenOracleAgent._preferred_handover_standoff(
            human_xy,
            start_xy,
        )
        occupancy = load_episode_occupancy(episode)
        planner = g1_navigation_planner(occupancy)
        if GoldenOracleAgent._handover_standoff_is_usable(
            planner,
            start_xy=start_xy,
            goal_xy=preferred,
            human_xy=human_xy,
        ):
            return preferred
        candidates = reachable_radial_occupancy_points(
            occupancy,
            center_xy=human_xy,
            reference_xy=start_xy,
            minimum_distance_m=(
                _HANDOVER_STANDOFF_M - _HANDOVER_STANDOFF_SAMPLE_SLACK_M
            ),
            maximum_distance_m=(
                _HANDOVER_STANDOFF_M + _HANDOVER_STANDOFF_SAMPLE_SLACK_M
            ),
            radial_spacing_m=0.02,
            angular_samples=72,
            clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        )
        for candidate in sorted(
            candidates,
            key=lambda point: math.dist(point, preferred),
        ):
            if GoldenOracleAgent._handover_standoff_is_usable(
                planner,
                start_xy=start_xy,
                goal_xy=candidate,
                human_xy=human_xy,
            ):
                return candidate
        raise ValueError(
            "Golden Oracle cannot author a G1-clear handover standoff "
            f"inside the legal band around {human_xy!r} from {start_xy!r}."
        )

    @staticmethod
    def _compiled_handover_phases(
        episode: Any,
        *,
        approach_from_xy: Sequence[float],
    ) -> list[dict[str, Any]]:
        human = GoldenOracleAgent._handover_recipient(episode)
        human_xy = GoldenOracleAgent._authored_human_xy(human)
        standoff = GoldenOracleAgent._reachable_handover_standoff(
            episode,
            human_xy=human_xy,
            approach_from_xy=approach_from_xy,
        )
        return [
            GoldenOracleAgent._nav_phase(
                standoff,
                "handover standoff",
                tolerance_m=_HANDOVER_STANDOFF_TOLERANCE_M,
                live_actor_clearance=True,
            ),
            {
                "kind": "handover",
                "target": str(human.name),
                "human_id": str(human.id),
                "reason": (
                    f"Offer the held object to {human.name} inside the "
                    "legal handover band."
                ),
            },
        ]

    @staticmethod
    def _compiled_grasp_phases(
        grasp: dict[str, Any],
        *,
        target_id: str,
        target: dict[str, Any],
        public_skill: bool = False,
    ) -> list[dict[str, Any]]:
        approach = GoldenOracleAgent._precise_phase(
            grasp["robot_stance_xy"],
            "collision-safe target grasp stance",
            tolerance_m=_G1_MANIPULATION_STANCE_TOLERANCE_M,
        )
        grasp_alignment = GoldenOracleAgent._orient_phase(
            target,
            stance=grasp["robot_stance_xy"],
            label="target grasp alignment",
            stance_tolerance_m=_G1_MANIPULATION_STANCE_TOLERANCE_M,
        )
        target_bounds_max = target.get("bounds_max_xyz")
        if (
            not isinstance(target_bounds_max, (list, tuple))
            or len(target_bounds_max) != 3
        ):
            raise ValueError(f"Grasp target {target_id!r} has no valid upper bounds.")
        crouched_grasp = float(target_bounds_max[2]) <= G1_CROUCH_GRASP_MAX_TARGET_TOP_M
        phases = [
            approach,
            grasp_alignment,
        ]
        # Public GraspGenX+cuRobo already plans pelvis height. A locomotion
        # crouch would lock stepping before the planner has used that DOF.
        # The compiled private-IK path still crouches on low furniture.
        if crouched_grasp and not public_skill:
            phases.append(
                {
                    "kind": "body_posture",
                    "action": "crouch",
                    "pelvis_height_m": G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
                    "reason": (
                        "Lower the official G1 locomotion posture at the authored "
                        "grasp stance before reaching the low target."
                    ),
                }
            )
        grasp_reach = {
            "kind": "grasp_reach",
            "target": target_id,
        }
        if public_skill:
            phrase = str(target.get("name") or target.get("category") or "").strip()
            if not phrase:
                raise ValueError(
                    f"Public grasp skill requires a visual name for {target_id!r}."
                )
            grasp_reach["public_skill"] = True
            grasp_reach["target_phrase"] = phrase
            grasp_reach["target_category"] = str(target.get("category") or "").strip()
            grasp_reach["certification_gate"] = "A"
            phases.append(grasp_reach)
            return phases
        phases.extend(
            [
                grasp_reach,
                {
                    "kind": "wait",
                    "remaining": 2,
                    "reason": "Verify the contact-gated physical target attachment.",
                },
            ]
        )
        for remaining_fraction in G1_CARRY_LOCOMOTION_UPPER_BODY_INTERMEDIATE_FRACTIONS:
            phases.append(
                {
                    "kind": "arm_pose",
                    "action": "prepare_carry",
                    "target": target_id,
                    "hand": "carried",
                    "locomotion_upper_body_remaining_fraction": (remaining_fraction),
                    "reason": (
                        "Retract both arms from transient grasp IK before locomotion."
                    ),
                }
            )
        phases.extend(
            [
                {
                    "kind": "arm_pose",
                    "action": "prepare_carry",
                    "target": target_id,
                    "hand": "carried",
                    "upper_command": [0.0] * 15,
                    "reset_to_locomotion_upper_body": True,
                    "reason": (
                        "Restore the official G1 locomotion upper-body posture."
                    ),
                },
                *(
                    [
                        {
                            "kind": "body_posture",
                            "action": "stand_up",
                            "reason": (
                                "Return to the certified standing posture after the "
                                "low-target grasp has been retracted."
                            ),
                        }
                    ]
                    if crouched_grasp
                    else []
                ),
                {
                    "kind": "back_away",
                    "windows": 2,
                    "speed_mps": 0.5,
                    "label": "source support after grasp",
                    "certification_gate": "A",
                },
            ]
        )
        return phases

    @staticmethod
    def _compiled_place_phases(
        place: dict[str, Any],
        *,
        target_id: str,
        target: dict[str, Any],
    ) -> list[dict[str, Any]]:
        support_surface = [float(value) for value in place["support_surface_xyz"]]
        settle_waits = max(1, int(place.get("settle_waits", 3)))
        place_ik = {
            "hand": "auto",
            "object_id": target_id,
            "support_surface_world": support_surface,
            "clearance_m": G1_PRE_RELEASE_CLEARANCE_M,
            "position_tolerance_m": (G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M),
            "posture": "front_place",
            "max_facing_error_deg": 30.0,
        }
        approach = GoldenOracleAgent._precise_phase(
            place["robot_stance_xy"],
            "collision-safe destination placement stance",
            tolerance_m=_G1_PLACE_STANCE_TOLERANCE_M,
        )
        crouched_place = float(support_surface[2]) <= G1_CROUCH_GRASP_MAX_TARGET_TOP_M
        # Position and facing are distinct causal phases. Reaching the stance
        # must happen first: if the robot faces the table while the authored
        # stance lies behind it, a forward locomotion policy cannot reliably
        # recover that position under payload. Once the base is at the stance,
        # orient toward the support surface and hand control to live carried-
        # object geometry.
        phases: list[dict[str, Any]] = [
            approach,
            GoldenOracleAgent._orient_phase(
                target,
                stance=place["robot_stance_xy"],
                target_xy=support_surface,
                label="destination placement alignment",
                stance_tolerance_m=_G1_PLACE_STANCE_TOLERANCE_M,
            ),
        ]
        placement_envelope = GoldenOracleAgent._precise_phase(
            place["robot_stance_xy"],
            "reachable destination placement envelope",
            tolerance_m=_G1_PLACE_STANCE_TOLERANCE_M,
        )
        placement_envelope["completion_place_ik"] = dict(place_ik)
        phases.append(placement_envelope)
        if crouched_place:
            phases.append(
                {
                    "kind": "body_posture",
                    "action": "crouch",
                    "pelvis_height_m": G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
                    "reason": (
                        "Lower the official G1 locomotion posture at the authored "
                        "placement stance. The gait is standing-only; all base "
                        "motion finishes before this crouch."
                    ),
                }
            )
        phases.extend(
            [
                {
                    "kind": "release",
                    "target": target_id,
                    "hand": "auto",
                    "certification_gate": "C",
                    "place_ik": place_ik,
                },
                {
                    "kind": "wait",
                    "remaining": settle_waits,
                    "reason": "Measure post-release placement stability.",
                },
            ]
        )
        if crouched_place:
            phases.append(
                {
                    "kind": "body_posture",
                    "action": "stand_up",
                    "reason": (
                        "Return to the certified standing posture after stable "
                        "placement."
                    ),
                }
            )
        phases.append({"kind": "report"})
        return phases

    @staticmethod
    def _nav_phase(
        goal: Any,
        label: str,
        tolerance_m: float = NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M,
        *,
        live_actor_clearance: bool = False,
    ) -> dict[str, Any]:
        phase = {
            "kind": "navigate",
            "goal": [float(goal[0]), float(goal[1])],
            "label": label,
            "tolerance_m": float(tolerance_m),
        }
        if live_actor_clearance:
            phase["live_actor_clearance"] = True
        return phase

    @staticmethod
    def _precise_phase(
        goal: Any, label: str, tolerance_m: float = 0.08
    ) -> dict[str, Any]:
        return {
            "kind": "precise_approach",
            "goal": [float(goal[0]), float(goal[1])],
            "label": label,
            "tolerance_m": float(tolerance_m),
        }

    @staticmethod
    def _orient_phase(
        obj: dict[str, Any],
        *,
        stance: Any,
        label: str,
        stance_tolerance_m: float,
        target_xy: Any | None = None,
        semantic_target: str | None = None,
    ) -> dict[str, Any]:
        if target_xy is None:
            lower = [float(value) for value in obj["bounds_min_xyz"]]
            upper = [float(value) for value in obj["bounds_max_xyz"]]
            center = [(lower[index] + upper[index]) * 0.5 for index in range(2)]
        else:
            if not isinstance(target_xy, (list, tuple)) or len(target_xy) < 2:
                raise ValueError("Authored manipulation target_xy must contain XY.")
            center = [float(target_xy[0]), float(target_xy[1])]
        target_angle = math.atan2(
            center[1] - float(stance[1]),
            center[0] - float(stance[0]),
        )
        phase = {
            "kind": "orient",
            "stance_xy": [float(stance[0]), float(stance[1])],
            "stance_tolerance_m": float(stance_tolerance_m),
            "yaw_rad": target_angle,
            "target_xy": center,
            "label": label,
            "tolerance_rad": 0.12,
        }
        if semantic_target:
            phase["semantic_target"] = str(semantic_target)
        return phase


__all__ = [
    "GoldenOracleAgent",
    "episode_success_requires_handover",
    "episode_success_requires_retained_object",
]
