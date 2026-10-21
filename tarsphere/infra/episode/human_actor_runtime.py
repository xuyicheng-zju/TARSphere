"""Milestone-driven human actor runtime for benchmark NPCs.

This runtime treats humans as benchmark actors, not evaluated embodied agents.
It consumes episode-level ``world_knowledge`` and milestone conditions, produces
structured human actions, and lets the existing Isaac/AnimPeople executor render
the visible body motion.
"""

from __future__ import annotations

from tarsphere.evaluation.engine import is_internal_task_phase

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional

from tarsphere.api_usage import aggregate_model_calls, merge_usage_reports
from tarsphere.bench.schema import TARSphereEpisode, HumanSpec
from tarsphere.evaluation.schema import MilestoneCondition, MilestoneReference
from tarsphere.evaluation.task_bindings import TaskBindings
from tarsphere.humans.runtime_capabilities import HUMAN_RUNTIME_CAPABILITIES
from tarsphere.infra.episode.interaction_backend import (
    DeterministicInteractionBackend,
)
from tarsphere.infra.episode.disruption_runtime import DisruptionRuntime
from tarsphere.contracts.human_interaction import (
    HUMAN_COMMUNICATION_RADIUS_M,
)

SUPPORTED_MILESTONE_COMMANDS = frozenset(HUMAN_RUNTIME_CAPABILITIES.command_names)
# Speech/symbolic work and idle control finish on application. Every other
# body command is asynchronous and must wait for the executor's success.
ONE_SHOT_COMMANDS = frozenset(
    item.name
    for item in HUMAN_RUNTIME_CAPABILITIES.commands
    if item.backend_command in {"", "Idle"}
)

SUPPORTED_TRIGGER_TYPES = {
    "causal references complete",
    "agent question",
    "agent action",
    "robot near human",
    "human visible",
    "robot near position",
    "milestone condition",
}


def _normalize_key(value: Any) -> str:
    return " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def _is_point(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        [float(item) for item in value[:2]]
    except (TypeError, ValueError):
        return False
    return True


def resolve_human_action_point(
    episode: TARSphereEpisode,
    value: Any,
) -> List[float]:
    """Resolve a Human action point from coordinates or world anchors."""
    if _is_point(value):
        point = [float(item) for item in list(value)[:3]]
        if len(point) == 2:
            point.append(0.0)
        return point
    target = _normalize_key(value)
    world = dict(getattr(episode, "world_knowledge", {}) or {})
    for section_name in ("anchors",):
        section = world.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, candidate in section.items():
            if _normalize_key(key) != target:
                continue
            if _is_point(candidate):
                point = [float(item) for item in list(candidate)[:3]]
                if len(point) == 2:
                    point.append(0.0)
                return point
            if isinstance(candidate, dict) and _is_point(candidate.get("center")):
                return [float(item) for item in list(candidate["center"])[:3]]
    raise ValueError(f"Could not resolve human action point: {value!r}")


@dataclass
class ActorState:
    human: HumanSpec
    queue: Deque[Dict[str, Any]] = field(default_factory=deque)
    active_action: Optional[Dict[str, Any]] = None
    # Dispatch alone cannot authorize completion from a previous body result.
    action_applied: bool = False
    active_command: str = ""
    active_target: Optional[List[float]] = None
    last_action: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """Observable outcome of one Agent question."""

    feedback: str
    human_id: str | None = None
    reply_message: str = ""
    response_mode: str = "unknown"
    # Authored route that produced this reply; unknown/pending replies have none.
    intent_id: str | None = None


class HumanActorRuntime:
    """Drive multiple human NPC actors from milestone conditions.

    This is the single Human actor runtime used by the episode runner.
    It schedules benchmark actor commands; the body executor renders those
    commands as AnimPeople motion in Isaac.
    """

    def __init__(
        self,
        episode: TARSphereEpisode,
    ):
        self.episode = episode
        self._world = dict(getattr(episode, "world_knowledge", {}) or {})
        self._humans = {str(human.id): human for human in episode.humans}
        self._human_aliases = self._build_human_aliases(episode.humans)
        self._interaction_backend = DeterministicInteractionBackend(episode)
        self._disruption = DisruptionRuntime(episode)
        self._validate_episode_contract()
        self._states: Dict[str, ActorState] = {}
        self._llm_policies: Dict[str, Any] = {}
        self._dialogue_history: dict[str, list[dict[str, str]]] = {}
        self._triggered_milestones: set[str] = set()
        # Public delivery opens reaction windows; queue admission does not.
        self._occurred_milestones: set[str] = set()
        self._completed_milestone_ids: set[str] = set()
        self._latest_condition_values: Dict[str, bool] = {}
        self._published_task_updates: set[tuple[str, str]] = set()
        self._public_task_values: Dict[str, str] = {}
        self.task_bindings = (
            TaskBindings(episode.evaluation) if episode.evaluation else None
        )
        self._public_speech_by_human: Dict[str, List[str]] = {}
        self._last_agent_question: Optional[Dict[str, Any]] = None
        self._last_agent_action: Optional[Dict[str, Any]] = None
        self._completed_human_movements: set[tuple[str, str]] = set()
        self._last_robot_action = ""
        self._last_robot_message = ""
        self._task_disclosed = False
        self._completed = False
        self._turns = 0
        self._last_step = -1
        self._replay_log: List[Dict[str, Any]] = []

    @property
    def human_ids(self) -> List[str]:
        return list(self._humans)

    @property
    def name(self) -> str:
        return self.human_ids[0] if self.human_ids else "human"

    @property
    def turns(self) -> int:
        return self._turns

    @property
    def task_disclosed(self) -> bool:
        return self._task_disclosed

    @property
    def completed(self) -> bool:
        return self._completed

    def reset(self):
        self._disruption.reset()
        self._states = {
            human_id: ActorState(human=human)
            for human_id, human in self._humans.items()
        }
        self._llm_policies.clear()
        self._dialogue_history.clear()
        self._triggered_milestones.clear()
        self._occurred_milestones.clear()
        self._completed_milestone_ids.clear()
        self._latest_condition_values.clear()
        self._published_task_updates.clear()
        self._public_task_values.clear()
        if self.task_bindings is not None:
            self.task_bindings.reset()
        initialized_fields: set[str] = set()
        for node in self._milestones():
            for update in node.event.task_updates:
                if update.field not in initialized_fields:
                    initialized_fields.add(update.field)
                    if update.previous_value_ref is not None:
                        self._public_task_values[update.field] = (
                            update.previous_value_ref
                        )
        self._public_speech_by_human.clear()
        self._last_agent_question = None
        self._last_agent_action = None
        self._completed_human_movements.clear()
        self._last_robot_action = ""
        self._last_robot_message = ""
        self._task_disclosed = False
        self._completed = False
        self._turns = 0
        self._last_step = -1
        self._replay_log.clear()
        if not self._humans:
            # In a Human-free component task the public system instruction is
            # the task disclosure. Seed that causal event at episode start so
            # Execution windows use the same semantic anchor as Human-authored
            # episodes without inventing a synthetic person or milestone.
            self._task_disclosed = True
            self._replay_log.append(
                {
                    "step": 0,
                    "event": "task_disclosed",
                    "source": "episode_system_prompt",
                }
            )

    def tick(
        self,
        scene_context: str = "",
        robot_action: Optional[Dict[str, Any]] = None,
        robot_message: Optional[str] = None,
        human_observations: Optional[Dict[str, Dict[str, Any]]] = None,
        step: int = 0,
        world_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        del scene_context
        human_observations = human_observations or {}
        snapshot = self._merge_snapshot(world_snapshot, human_observations, step)
        self._observe_robot_event(robot_action, robot_message, step=step)
        self._refresh_active_actions(snapshot)
        # The runner may tick Humans again after a robot utterance in the same
        # decision step. New milestones belong to the next pre-agent tick:
        # firing them post-decision would timestamp an event before the agent
        # could observe or answer it.
        pre_agent_tick = step != self._last_step
        if pre_agent_tick:
            self._enqueue_triggered_milestone(snapshot)
            self._last_step = step
        return self._schedule_actions(snapshot, open_events=pre_agent_tick)

    def observe_milestone_conditions(self, values: Dict[str, bool]) -> None:
        """Publish the runner's latest primitive samples to the event runtime."""

        self._latest_condition_values = {
            str(condition_id): bool(value) for condition_id, value in values.items()
        }

    def observe_completed_milestones(self, milestone_ids: Iterable[str]) -> None:
        """Publish the monotone evaluator-confirmed reaction set."""

        value = {str(item) for item in milestone_ids}
        known = {str(item.milestone_id) for item in self._milestones()}
        unknown = sorted(value - known)
        if unknown:
            raise ValueError(
                f"completed milestone set references unknown ids: {unknown}"
            )
        if not self._completed_milestone_ids.issubset(value):
            raise ValueError("completed milestone set cannot move backward")
        self._completed_milestone_ids = value

    def mark_prepared_milestone_completed(
        self,
        *,
        milestone_id: str,
        human_id: str,
        step: int,
        start_physics_step: Optional[int] = None,
        physics_step: Optional[int] = None,
    ) -> None:
        """Record a physical Human setup completed before policy step zero.

        The environment uses this only for the spatial blocker's real pre-roll
        from its collision-safe spawn to the authored choke point.  Recording
        it here prevents the normal event scheduler from issuing the same
        ``goto`` again after the task plan is submitted.
        """

        profile = getattr(self.episode.evaluation, "task_profile", None)
        if (
            getattr(profile, "subtype", None) != "spatial_blocking"
            or str(getattr(profile, "blocking_milestone_id", "")) != str(milestone_id)
            or str(getattr(profile, "blocking_human_id", "")) != str(human_id)
        ):
            raise ValueError(
                "Only the configured spatial blocking event may be prepared."
            )
        known = {str(item.milestone_id): item for item in self._milestones()}
        milestone = known.get(str(milestone_id))
        if milestone is None:
            raise ValueError(f"Unknown prepared milestone {milestone_id!r}.")
        motions = [
            action
            for action in milestone.event.human_actions
            if (
                action.get("command") == "goto"
                and str(action.get("human") or "") == str(human_id)
            )
        ]
        if len(motions) != 1:
            raise ValueError(
                "Prepared spatial blocking event must own one matching goto."
            )
        self._triggered_milestones.add(str(milestone_id))
        self._completed_human_movements.add((str(human_id), str(milestone_id)))
        self._replay_log.extend(
            (
                {
                    "step": int(step),
                    "physics_step": start_physics_step,
                    "event": "milestone_triggered",
                    "milestone_id": str(milestone_id),
                    "trigger": self._milestone_trigger(milestone),
                    "queued_actions": len(motions),
                    "phase": "episode_setup",
                },
                {
                    "step": int(step),
                    "physics_step": start_physics_step,
                    "event": "human_action_applied",
                    "human_id": str(human_id),
                    "milestone_id": str(milestone_id),
                    "command": "goto",
                    "message": "",
                    "phase": "episode_setup",
                },
                {
                    "step": int(step),
                    "physics_step": physics_step,
                    "event": "human_motion_ended",
                    "human_id": str(human_id),
                    "milestone_id": str(milestone_id),
                    "phase": "episode_setup",
                },
            )
        )
        self._record_milestone_occurrence(
            str(milestone_id),
            {
                "step": int(step),
                "physics_steps": physics_step,
            },
        )

    def _milestones(self) -> tuple[Any, ...]:
        evaluation = getattr(self.episode, "evaluation", None)
        return tuple(getattr(evaluation, "milestones", ()) or ())

    @staticmethod
    def _condition_trigger(condition: Any) -> Dict[str, Any]:
        detector = str(condition.detector)
        arguments = dict(condition.arguments)
        if detector == "robot_near_human":
            return {
                "type": "robot near human",
                "human": arguments["human_id"],
                "distance_m": arguments["distance"],
            }
        if detector == "human_visible":
            return {
                "type": "human visible",
                "human": arguments["human_id"],
                "condition_id": str(condition.condition_id),
            }
        if detector == "robot_in_place":
            return {
                "type": "robot near position",
                "position": arguments["position"],
                "distance_m": arguments["distance"],
            }
        if detector == "agent_interaction_selected":
            return {
                "type": "agent question",
                "human": arguments["human_id"],
                "intent_id": arguments["intent_id"],
            }
        if detector == "agent_action_selected":
            return {"type": "agent action", "action": arguments["action"]}
        return {
            "type": "milestone condition",
            "condition_id": str(condition.condition_id),
        }

    def _milestone_trigger(self, milestone: Any) -> Dict[str, Any]:
        conditions = [
            item
            for item in milestone.preconditions
            if isinstance(item, MilestoneCondition)
        ]
        triggers = [self._condition_trigger(item) for item in conditions]
        if not triggers:
            return {"type": "causal references complete"}
        return triggers[0] if len(triggers) == 1 else {"all": triggers}

    def get_human_state(self, human_id: str) -> Dict[str, Any]:
        state = self._states.get(self._resolve_human_id(human_id))
        if state is None:
            return {}
        return {
            "active_command": state.active_command,
            "active_target": state.active_target,
            "queued_actions": len(state.queue),
            "last_action": state.last_action,
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        return {human_id: self.get_human_state(human_id) for human_id in self.human_ids}

    def get_replay_log(self) -> List[Dict[str, Any]]:
        return list(self._replay_log)

    def event_actions_completed(self, milestone_id: str) -> bool:
        """All authored actions finished, not merely admitted to an actor queue."""
        if milestone_id not in self._occurred_milestones:
            return False
        return not any(
            (
                state.active_action is not None
                and state.active_action.get("milestone_id") == milestone_id
            )
            or any(action.get("milestone_id") == milestone_id for action in state.queue)
            for state in self._states.values()
        )

    def acknowledge_action_applied(
        self,
        human_id: str,
        action: Dict[str, Any],
        *,
        step: int,
        physics_step: Optional[int] = None,
    ) -> None:
        """Complete a synchronous command after the environment applied it.

        Queue dispatch is not execution. Conversely, delivered speech must
        not remain active until another Robot decision happens.
        """
        state = self._states[self._resolve_human_id(human_id)]
        if state.active_action != action:
            raise RuntimeError("Applied Human action does not match its active command")
        if state.action_applied:
            raise RuntimeError("Human action was already acknowledged")
        state.action_applied = True
        snapshot = {"step": step, "physics_steps": physics_step}
        self._replay_log.append(
            {
                **self._event_clock(snapshot),
                "event": "human_action_applied",
                "human_id": human_id,
                "command": action["command"],
                "message": str(action.get("message") or ""),
                "milestone_id": action["milestone_id"],
            }
        )
        milestone_id = str(action["milestone_id"])
        milestone = next(
            item for item in self._milestones() if item.milestone_id == milestone_id
        )
        speaking = {"speak", "ask", "report_done"}
        arrival_publishes_event = self._is_spatial_clearance_movement(
            milestone_id,
            action,
        )
        if not arrival_publishes_event and (
            action["command"] in speaking
            or not any(
                item.get("command") in speaking
                for item in milestone.event.human_actions
            )
        ):
            self._record_milestone_occurrence(milestone_id, snapshot)
        message = str(action.get("message") or "").strip()
        if message and action["command"] in speaking:
            self._public_speech_by_human.setdefault(human_id, []).append(message)
            if human_id == self.human_ids[0]:
                self._task_disclosed = True
            self._publish_task_updates(
                milestone_id=milestone_id,
                source_human_id=human_id,
                step=step,
                physics_step=physics_step,
            )
        if action["command"] not in ONE_SHOT_COMMANDS:
            return
        self._complete_active_action(human_id, state, snapshot, "action applied")

    def bind_spatial_clearance_route(
        self,
        human_id: str,
        action: Dict[str, Any],
        *,
        target: Iterable[float],
        waypoints: Iterable[Iterable[float]],
        step: int,
        robot_position: Iterable[float],
        physics_step: Optional[int] = None,
    ) -> None:
        """Replace the authored clearance witness with one live A* route."""

        state = self._states[self._resolve_human_id(human_id)]
        if state.active_action is not action:
            raise RuntimeError(
                "Dynamic Human route does not match the active clearance action"
            )
        resolved_target = [float(value) for value in target]
        resolved_waypoints = [[float(value) for value in point] for point in waypoints]
        if not resolved_waypoints or resolved_waypoints[-1] != resolved_target:
            raise ValueError(
                "Dynamic spatial-clearance route must end at its selected target"
            )
        action["target"] = resolved_target
        action["waypoints"] = resolved_waypoints
        action["route_source"] = "live_robot_aware_astar"
        state.active_target = resolved_target
        self._replay_log.append(
            {
                "step": int(step),
                "physics_step": physics_step,
                "event": "spatial_clearance_route_bound",
                "human_id": str(human_id),
                "milestone_id": action.get("milestone_id"),
                "robot_position": [float(value) for value in robot_position],
                "target": list(resolved_target),
                "waypoints": [list(point) for point in resolved_waypoints],
            }
        )

    def defer_spatial_clearance_action(
        self,
        human_id: str,
        action: Dict[str, Any],
        *,
        step: int,
        reason: str,
        physics_step: Optional[int] = None,
    ) -> None:
        """Keep the blocker still and retry the same clear event next turn."""

        state = self._states[self._resolve_human_id(human_id)]
        if state.active_action is not action:
            raise RuntimeError(
                "Deferred Human route does not match the active clearance action"
            )
        state.queue.appendleft(action)
        state.active_action = None
        state.action_applied = False
        state.active_command = ""
        state.active_target = None
        self._replay_log.append(
            {
                "step": int(step),
                "physics_step": physics_step,
                "event": "spatial_clearance_waiting_for_robot",
                "human_id": str(human_id),
                "milestone_id": action.get("milestone_id"),
                "reason": str(reason),
            }
        )

    def record_speech_delivery(
        self,
        human_id: str,
        message: str,
        *,
        step: int,
        physics_step: Optional[int] = None,
    ) -> None:
        """Record an interaction reply when the environment makes it public.

        Generating a question answer is separate from delivering it. This
        receipt does not publish a milestone or change dialogue history.
        """
        human_id = self._resolve_human_id(human_id)
        message = str(message).strip()
        if not message:
            raise ValueError("Delivered Human speech cannot be empty")
        self._replay_log.append(
            {
                "step": int(step),
                "physics_step": physics_step,
                "event": "human_utterance_delivered",
                "human_id": human_id,
                "message": message,
            }
        )

    def model_usage(self) -> Dict[str, Any]:
        """Report NPC language usage separately from the evaluated Agent."""

        reports = [policy.model_usage() for policy in self._llm_policies.values()]
        if not reports:
            return aggregate_model_calls(())
        return merge_usage_reports(reports)

    def question_feedback(
        self,
        *,
        step: int,
        human_observations: Dict[str, Dict[str, Any]],
        world_snapshot: Dict[str, Any],
    ) -> QuestionOutcome | None:
        """Describe the observable outcome of the current ``ask`` action.

        Milestone answers remain one-step delayed, but an attempted question
        must never disappear into silence.  The feedback names only what the
        robot could learn from the exchange; it does not disclose the expected
        recipient, a hidden answer, or an unreached milestone id.
        """

        question = self._last_agent_question
        if question is None or int(question["step"]) != int(step):
            return None
        recipient_name = str(question["recipient"])
        recipient_id = self._resolve_human_id(recipient_name)
        if recipient_id not in self._humans:
            return QuestionOutcome(
                feedback=(
                    f"No known person named {recipient_name!r} could be addressed."
                ),
                response_mode="unknown_recipient",
            )

        snapshot = self._merge_snapshot(
            world_snapshot,
            human_observations,
            step,
        )
        recipient_state = (snapshot.get("humans") or {}).get(recipient_id) or {}
        if recipient_state.get("robot_can_communicate") is not True:
            return QuestionOutcome(
                feedback=(
                    f"You called to {recipient_name}, but no reply was heard. "
                    "Move closer and bring them into view before trying again."
                ),
                human_id=recipient_id,
                response_mode="not_communicable",
            )

        physics_step = snapshot.get("physics_steps")
        if physics_step is None and self._disruption.profile is not None:
            raise RuntimeError("Human disruption requires the simulator physics clock")
        unavailable = self._disruption.unavailable_response(
            recipient_id,
            int(physics_step or 0),
        )
        if unavailable is not None:
            return self._spoken_question_outcome(
                human_id=recipient_id,
                message=unavailable,
                response_mode="temporarily_unavailable",
                step=step,
            )

        pending = []
        milestones = self._milestones()
        for milestone in milestones:
            was_delivered = milestone.milestone_id in self._occurred_milestones
            is_frontier = not was_delivered and self._milestone_references_complete(
                milestone
            )
            trigger = self._milestone_trigger(milestone)
            leaves = self._trigger_leaves(trigger)
            question_triggers = [
                item
                for item in leaves
                if _normalize_key(item.get("type") or item.get("when") or "")
                == "agent question"
            ]
            if not question_triggers:
                continue
            if is_frontier:
                pending.append((milestone, leaves, question_triggers[0]))

        # Every causally open node may answer. Ambiguous routes at the same DAG
        # frontier are an authoring error, never resolved by declaration order.
        frontier = pending
        matching = [
            item
            for item in frontier
            if self._question_trigger_matches_identity(item[2], question)
        ]
        if len(matching) > 1:
            raise ValueError(
                "Episode has ambiguous NPC responses for one "
                f"(recipient, intent) pair: {recipient_name!r}, "
                f"{question['intent_id']!r}"
            )
        if matching:
            _, leaves, _ = matching[0]
            next_snapshot = dict(snapshot)
            next_snapshot["step"] = int(step) + 1
            for condition in leaves:
                trigger_type = _normalize_key(
                    condition.get("type") or condition.get("when") or ""
                )
                if trigger_type == "agent question":
                    continue
                if self._trigger_matches(condition, next_snapshot):
                    continue
                if trigger_type == "robot near human":
                    return QuestionOutcome(
                        feedback=(
                            f"You called to {recipient_name}, but no reply "
                            "was heard. Move closer before trying again."
                        ),
                        human_id=recipient_id,
                        response_mode="not_communicable",
                    )
                if trigger_type == "human visible":
                    return QuestionOutcome(
                        feedback=(
                            f"You called to {recipient_name}, but no reply "
                            "was heard. Bring them into view before trying "
                            "again."
                        ),
                        human_id=recipient_id,
                        response_mode="not_visible",
                    )
                return self._unknown_answer(recipient_id, step=step)
            return QuestionOutcome(
                feedback=f"{recipient_name} heard your question and is responding.",
                human_id=recipient_id,
                response_mode="pending_milestone_response",
            )

        if not str(question["intent_id"]).strip():
            return self._natural_language_answer(
                recipient_id, question, snapshot, step=step
            )
        routed = self._interaction_backend.route(
            recipient_id=recipient_id,
            intent_id=str(question["intent_id"]),
            world_snapshot=snapshot,
            completed_milestones=frozenset(self._completed_milestone_ids),
        )
        if routed is not None:
            return self._spoken_question_outcome(
                human_id=recipient_id,
                message=routed.message,
                response_mode="accept",
                step=step,
                intent_id=routed.intent_id,
            )
        # A named-but-unroutable id (invented, mistyped, private, or gated by
        # a still-open precondition) must not end the exchange on its own:
        # weak instruction-following routinely fills a semantically plausible
        # id. The message text re-enters natural-language routing, which
        # re-applies every authorization gate before answering.
        return self._natural_language_answer(
            recipient_id, question, snapshot, step=step
        )

    def _natural_language_answer(
        self,
        human_id: str,
        question: dict[str, Any],
        snapshot: dict[str, Any],
        *,
        step: int,
    ) -> QuestionOutcome:
        topics = self._interaction_backend.available_topics(
            recipient_id=human_id,
            completed_milestones=frozenset(self._completed_milestone_ids),
        )
        human = self._humans[human_id]
        facts = [fact.model_dump() for fact in human.answer_facts]
        from tarsphere.humans.llm_policy import HumanDialoguePolicy

        policy = self._llm_policies.get(human_id)
        if policy is None:
            policy = HumanDialoguePolicy(
                npc_context={
                    "id": str(human.id),
                    "public_description": str(human.public_description),
                    "public_job": str(human.public_job),
                    "persona": str(human.persona),
                    "goals": [str(goal) for goal in human.goals],
                }
            )
            self._llm_policies[human_id] = policy
        history = self._dialogue_history.get(human_id, [])
        recent_human_utterances = self._public_speech_by_human.get(human_id, [])[-3:]
        text = str(question["message"])
        nearby_humans = []
        for nearby_id, state in (snapshot.get("humans") or {}).items():
            if not isinstance(state, dict) or not (
                state.get("human_visible_to_robot") is True
                or state.get("robot_can_communicate") is True
            ):
                continue
            spec = self._humans.get(str(nearby_id))
            if spec is None:
                continue
            nearby_humans.append(
                {
                    "id": str(spec.id),
                    "public_description": str(spec.public_description),
                    "public_job": str(spec.public_job),
                    "visible": state.get("human_visible_to_robot") is True,
                    "addressable": state.get("robot_can_communicate") is True,
                }
            )
        routing = policy.route_question(
            question=text,
            topics=topics,
            recent_human_utterances=recent_human_utterances,
            history=history,
            public_context={
                "recipient": {
                    "id": str(human.id),
                    "public_description": str(human.public_description),
                    "public_job": str(human.public_job),
                },
                "nearby_humans": nearby_humans,
                "recent_public_utterances": recent_human_utterances,
            },
        )
        selected = routing.intent_id
        self._replay_log.append(
            {
                "event": "question_routed",
                "step": step,
                "human_id": human_id,
                "question": text,
                "intent_id": selected,
                "authorized_topics": topics,
                "recent_human_utterances": recent_human_utterances,
            }
        )
        if selected is not None:
            # A second authorization check protects against invalid classifier output.
            routed = self._interaction_backend.route(
                recipient_id=human_id,
                intent_id=selected,
                world_snapshot=snapshot,
                completed_milestones=frozenset(self._completed_milestone_ids),
                allow_private=True,
            )
            if routed is None:
                raise ValueError("NPC classifier selected an unavailable topic")
            return self._spoken_question_outcome(
                human_id=human_id,
                message=routed.message,
                response_mode="accept",
                step=step,
                intent_id=routed.intent_id,
            )
        if facts:
            answer = policy.answer(question=text, facts=facts, history=history)
            self._replay_log.append(
                {
                    "event": "supplementary_answer_generated",
                    "step": step,
                    "human_id": human_id,
                    "question": text,
                    **answer.model_dump(),
                }
            )
            if answer.message.strip():
                return self._spoken_question_outcome(
                    human_id=human_id,
                    message=answer.message,
                    response_mode="supplementary",
                    step=step,
                )
        return self._unknown_answer(human_id, step=step)

    def _spoken_question_outcome(
        self,
        *,
        human_id: str,
        message: str,
        response_mode: str,
        step: int,
        intent_id: str | None = None,
    ) -> QuestionOutcome:
        reply = str(message).strip()
        if not reply:
            raise ValueError("A spoken Human question outcome cannot be empty")
        self._public_speech_by_human.setdefault(human_id, []).append(reply)
        record = {
            "step": int(step),
            "event": "question_answered",
            "human_id": human_id,
            "response_mode": response_mode,
            "message": reply,
        }
        if intent_id:
            record["intent_id"] = str(intent_id)
        self._replay_log.append(record)
        question = self._last_agent_question
        if question is not None and question["step"] == step:
            history = self._dialogue_history.setdefault(human_id, [])
            history.append({"question": str(question["message"]), "answer": reply})
            del history[:-3]
        return QuestionOutcome(
            feedback=f'{human_id} says: "{reply}"',
            human_id=human_id,
            reply_message=reply,
            response_mode=response_mode,
            intent_id=intent_id,
        )

    def _unknown_answer(self, recipient_name: str, *, step: int) -> QuestionOutcome:
        """Return one non-revealing answer for every invalid in-range question."""

        return self._spoken_question_outcome(
            human_id=recipient_name,
            message="I don't know about that.",
            response_mode="unknown",
            step=step,
        )

    @staticmethod
    def _trigger_leaves(trigger: Any) -> list[Dict[str, Any]]:
        if not isinstance(trigger, dict):
            raise TypeError(f"Milestone trigger must be a mapping: {trigger!r}")
        if "all" in trigger or "any" in trigger:
            key = "all" if "all" in trigger else "any"
            leaves: list[Dict[str, Any]] = []
            for item in trigger[key]:
                leaves.extend(HumanActorRuntime._trigger_leaves(item))
            return leaves
        return [dict(trigger)]

    def _question_trigger_matches_identity(
        self,
        trigger: Dict[str, Any],
        question: Dict[str, Any],
    ) -> bool:
        return (
            self._resolve_human_id(trigger.get("human") or "")
            == self._resolve_human_id(question["recipient"])
            and str(trigger.get("intent_id") or "").strip()
            == str(question["intent_id"]).strip()
        )

    def get_last_action(self, human_id: str) -> Optional[Dict[str, Any]]:
        state = self._states.get(self._resolve_human_id(human_id))
        return None if state is None else state.last_action

    def observe_robot_action(
        self,
        robot_action: Dict[str, Any],
        *,
        step: int,
    ) -> None:
        """Register one complete executed Robot decision exactly once."""

        if not isinstance(robot_action, dict) or not robot_action:
            raise TypeError(
                "observe_robot_action requires a non-empty decision mapping"
            )
        self._observe_robot_event(robot_action, None, step=step)

    def _observe_robot_event(
        self,
        robot_action: Optional[Dict[str, Any]],
        robot_message: Optional[str],
        *,
        step: int,
    ):
        if robot_action:
            self._last_robot_action = str(
                robot_action.get("action") or robot_action.get("name") or ""
            )
            self._last_agent_action = {
                "step": int(step),
                "action": self._last_robot_action,
                "recipient": str(robot_action.get("recipient") or "").strip(),
            }
            if self._last_robot_action == "report_done":
                self._completed = True
            if self._last_robot_action == "ask":
                recipient = str(robot_action.get("recipient") or "").strip()
                intent_id = str(robot_action.get("intent_id") or "").strip()
                if not recipient:
                    raise ValueError(
                        "ask requires non-empty recipient "
                        "before it can trigger a Human milestone."
                    )
                self._last_agent_question = {
                    "step": int(step),
                    "action": "ask",
                    "recipient": recipient,
                    "intent_id": intent_id,
                    "message": str(
                        robot_action.get("message") or robot_message or ""
                    ).strip(),
                }
        if robot_message:
            self._last_robot_message = str(robot_message).strip()

    @staticmethod
    def _build_human_aliases(humans: Iterable[HumanSpec]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for human in humans:
            runtime_id = str(human.id)
            aliases[runtime_id] = runtime_id
        return aliases

    def _resolve_human_id(self, value: Any) -> str:
        key = str(value)
        return self._human_aliases.get(key, key)

    def _merge_snapshot(
        self,
        world_snapshot: Optional[Dict[str, Any]],
        human_observations: Dict[str, Dict[str, Any]],
        step: int,
    ) -> Dict[str, Any]:
        snapshot = dict(world_snapshot or {})
        # The runner normalizes raw environment clocks to physics_steps.
        # A caller's decision step is never evidence of elapsed physics time.
        snapshot["step"] = step
        snapshot.setdefault("humans", {})
        for human_id, observation in human_observations.items():
            runtime_id = self._resolve_human_id(human_id)
            human_state = dict(snapshot["humans"].get(runtime_id, {}))
            if not human_state:
                human_state = dict(snapshot["humans"].get(str(human_id), {}))
            human_state.update(observation or {})
            snapshot["humans"][runtime_id] = human_state
        snapshot.setdefault("robot", {})
        if self._last_robot_action:
            snapshot["robot"]["last_action"] = self._last_robot_action
        if self._last_robot_message:
            snapshot["robot"]["last_message"] = self._last_robot_message
        return snapshot

    def _refresh_active_actions(self, snapshot: Dict[str, Any]):
        for human_id, state in self._states.items():
            if state.active_action is None or not state.action_applied:
                continue
            command = state.active_command
            if command not in ONE_SHOT_COMMANDS and self._body_action_completed(
                human_id, snapshot
            ):
                self._complete_active_action(
                    human_id,
                    state,
                    snapshot,
                    "goto target reached"
                    if command == "goto"
                    else "body command completed",
                )

    @staticmethod
    def _event_clock(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Decision step and observed simulator tick, never interchangeable.

        physics_step is null for adapters with no simulator clock. It is the
        tick at which this transition was observed/committed, not an estimate
        of the first visible motion frame inside an earlier Robot action.
        """
        return {
            "step": snapshot.get("step"),
            "physics_step": snapshot.get("physics_steps"),
        }

    def _complete_active_action(
        self,
        human_id: str,
        state: ActorState,
        snapshot: Dict[str, Any],
        reason: str,
    ):
        completed_action = dict(state.active_action or {})
        self._replay_log.append(
            {
                **self._event_clock(snapshot),
                "event": "human_action_completed",
                "human_id": human_id,
                "command": state.active_command,
                "milestone_id": completed_action.get("milestone_id"),
                "reason": reason,
            }
        )
        if state.active_command == "goto":
            milestone_id = str(completed_action.get("milestone_id") or "").strip()
            if not milestone_id:
                raise RuntimeError(
                    f"Completed Human GoTo for {human_id!r} has no milestone_id."
                )
            self._completed_human_movements.add((human_id, milestone_id))
            if self._is_spatial_clearance_movement(
                milestone_id,
                completed_action,
            ):
                self._record_milestone_occurrence(milestone_id, snapshot)
            self._replay_log.append(
                {
                    **self._event_clock(snapshot),
                    "event": "human_motion_ended",
                    "human_id": human_id,
                    "milestone_id": completed_action.get("milestone_id"),
                }
            )
        state.active_action = None
        state.action_applied = False
        state.active_command = ""
        state.active_target = None

    def _is_spatial_clearance_movement(
        self,
        milestone_id: str,
        action: Dict[str, Any],
    ) -> bool:
        profile = getattr(self.episode.evaluation, "task_profile", None)
        return bool(
            getattr(profile, "subtype", None) == "spatial_blocking"
            and str(getattr(profile, "clear_milestone_id", "")) == str(milestone_id)
            and str(action.get("command") or "") == "goto"
        )

    def _body_action_completed(
        self,
        human_id: str,
        snapshot: Dict[str, Any],
    ) -> bool:
        human_state = (snapshot.get("humans") or {}).get(human_id) or {}
        status = str(human_state.get("action_status") or "").lower()
        # Both motion backends own arrival. Proximity alone can precede route
        # completion and must not unlock a queued utterance or task revision.
        return status == "succeed" and not human_state.get("pose_stale", False)

    def _enqueue_triggered_milestone(self, snapshot: Dict[str, Any]):
        """Emit every newly eligible DAG event once per policy turn."""

        triggered_now = [
            milestone
            for milestone in self._milestones()
            if milestone.milestone_id not in self._triggered_milestones
            and not is_internal_task_phase(milestone)
            and self._milestone_references_complete(milestone)
            and self._trigger_matches(self._milestone_trigger(milestone), snapshot)
        ]

        # A milestone unlocked during this tick is visible to dependencies on
        # the next decision tick. This prevents an authored milestone chain
        # from collapsing into one frame merely because the declarations are
        # stored in topological order.
        for milestone in triggered_now:
            milestone_id = str(milestone.milestone_id)
            self._triggered_milestones.add(milestone_id)
            actions = [dict(action) for action in milestone.event.human_actions]
            for action in actions:
                human_id = self._resolve_human_id(
                    action.get("human") or action.get("actor") or ""
                )
                state = self._states.get(human_id)
                if state is None:
                    raise RuntimeError(
                        f"Milestone {milestone_id} resolved to unknown human "
                        f"{human_id!r}: {action!r}"
                    )
                action["milestone_id"] = milestone_id
                state.queue.append(action)

            self._replay_log.append(
                {
                    **self._event_clock(snapshot),
                    "event": "milestone_triggered",
                    "milestone_id": milestone_id,
                    "trigger": self._milestone_trigger(milestone),
                    "queued_actions": len(actions),
                }
            )
            if not actions:
                self._record_milestone_occurrence(milestone_id, snapshot)

    def _record_milestone_occurrence(
        self,
        milestone_id: str,
        snapshot: Dict[str, Any],
    ) -> None:
        """Timestamp one publicly delivered event, independently of its score."""
        if milestone_id in self._occurred_milestones:
            return
        self._occurred_milestones.add(milestone_id)
        self._replay_log.append(
            {
                **self._event_clock(snapshot),
                "event": "milestone_event_occurred",
                "milestone_id": milestone_id,
            }
        )

    def _milestone_references_complete(self, milestone: Any) -> bool:
        references = {
            item.milestone_ref
            for item in milestone.preconditions
            if isinstance(item, MilestoneReference)
        }
        return references.issubset(self._completed_milestone_ids)

    def _trigger_matches(
        self,
        trigger: Any,
        snapshot: Dict[str, Any],
    ) -> bool:
        if trigger is None:
            raise ValueError("Human milestone trigger is missing.")
        if isinstance(trigger, str):
            trigger_type = _normalize_key(trigger)
            trigger_data: Dict[str, Any] = {}
        elif isinstance(trigger, dict):
            if "all" in trigger:
                return all(
                    self._trigger_matches(
                        item,
                        snapshot,
                    )
                    for item in trigger["all"]
                )
            if "any" in trigger:
                return any(
                    self._trigger_matches(
                        item,
                        snapshot,
                    )
                    for item in trigger["any"]
                )
            trigger_type = _normalize_key(
                trigger.get("type") or trigger.get("when") or ""
            )
            trigger_data = trigger
        else:
            raise TypeError(f"Unsupported human milestone trigger object: {trigger!r}")

        if trigger_type == "causal references complete":
            return True
        if trigger_type == "agent question":
            return self._agent_question_matches(
                trigger_data,
                snapshot,
            )
        if trigger_type == "agent action":
            return self._agent_action_matches(trigger_data, snapshot)
        if trigger_type == "milestone condition":
            condition_id = str(trigger_data.get("condition_id") or "").strip()
            if not condition_id:
                raise ValueError("milestone condition trigger requires condition_id")
            return bool(self._latest_condition_values.get(condition_id, False))
        if trigger_type == "robot near human":
            return self._robot_near_human(trigger_data, snapshot)
        if trigger_type == "human visible":
            human_id = self._resolve_human_id(trigger_data.get("human") or "")
            human_state = (snapshot.get("humans") or {}).get(human_id) or {}
            if "human_visible_to_robot" in human_state:
                return bool(human_state["human_visible_to_robot"])
            condition_id = str(trigger_data.get("condition_id") or "").strip()
            if not condition_id:
                raise ValueError("human visible trigger requires condition_id")
            return bool(self._latest_condition_values.get(condition_id, False))
        if trigger_type == "robot near position":
            return self._robot_near_position(trigger_data, snapshot)
        raise ValueError(f"Unsupported human milestone trigger type: {trigger_type!r}")

    def _agent_question_matches(
        self,
        trigger: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> bool:
        question = self._last_agent_question
        if question is None:
            return False
        try:
            current_step = int(snapshot["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Agent question trigger requires the current decision "
                "step in the world snapshot."
            ) from exc
        if int(question["step"]) != current_step - 1:
            return False

        expected_human = self._resolve_human_id(trigger.get("human") or "")
        actual_human = self._resolve_human_id(question["recipient"])
        if actual_human != expected_human:
            return False
        if (
            str(question["intent_id"]).strip()
            != str(trigger.get("intent_id") or "").strip()
        ):
            return False
        return True

    def _agent_action_matches(
        self,
        trigger: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> bool:
        action = self._last_agent_action
        if action is None:
            return False
        try:
            current_step = int(snapshot["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Agent action trigger requires the current decision step in "
                "the world snapshot."
            ) from exc
        if int(action["step"]) != current_step - 1:
            return False
        if _normalize_key(action["action"]) != _normalize_key(
            trigger.get("action") or ""
        ):
            return False
        expected_human = str(trigger.get("human") or "").strip()
        if not expected_human:
            return True
        return self._resolve_human_id(action["recipient"]) == self._resolve_human_id(
            expected_human
        )

    def _public_speech_observed(self, trigger: Dict[str, Any]) -> bool:
        human_id = self._resolve_human_id(trigger.get("human") or "")
        messages = self._public_speech_by_human.get(human_id) or []
        if not messages:
            return False
        contains = (
            trigger.get("contains")
            or trigger.get("message_contains")
            or trigger.get("text_contains")
        )
        if contains in (None, ""):
            return True
        needle = str(contains).strip().lower()
        return any(needle in str(message).lower() for message in messages)

    def _robot_near_human(
        self, trigger: Dict[str, Any], snapshot: Dict[str, Any]
    ) -> bool:
        human_id = self._resolve_human_id(trigger.get("human") or "")
        threshold = float(trigger["distance_m"])
        human_state = (snapshot.get("humans") or {}).get(human_id) or {}
        distance = human_state.get("robot_distance_m")
        if distance is not None:
            try:
                return float(distance) <= threshold
            except (TypeError, ValueError):
                return False
        robot_position = (snapshot.get("robot") or {}).get("position")
        human_position = human_state.get("position")
        if not (_is_point(robot_position) and _is_point(human_position)):
            return False
        return self._distance_xy(robot_position, human_position) <= threshold

    def _robot_near_position(
        self,
        trigger: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> bool:
        position = (snapshot.get("robot") or {}).get("position")
        if not _is_point(position):
            return False
        return self._distance_xy(position, trigger["position"]) <= float(
            trigger["distance_m"]
        )

    def _schedule_actions(
        self,
        snapshot: Dict[str, Any],
        *,
        open_events: bool,
    ) -> Dict[str, Dict[str, Any]]:
        scheduled: Dict[str, Dict[str, Any]] = {}
        for human_id, state in self._states.items():
            if state.active_action is not None or not state.queue:
                continue
            raw_action = state.queue[0]
            milestone_id = str(raw_action["milestone_id"])
            event_pending = milestone_id not in self._occurred_milestones
            # Admission consumes the trigger. Agent questions and actions are
            # one-step events and must survive while this actor is busy with an
            # earlier action. Delivery still waits for a pre-Agent tick and,
            # for speech, live communication range.
            if event_pending and not open_events:
                continue
            if str(raw_action["command"]) in {"speak", "ask", "report_done"}:
                # A queued utterance is audible only at its actual dispatch.
                # Human-to-Robot speech deliberately has no camera gate.
                if not self._robot_near_human(
                    {
                        "human": human_id,
                        "distance_m": HUMAN_COMMUNICATION_RADIUS_M,
                    },
                    snapshot,
                ):
                    continue
            state.queue.popleft()
            action = self._prepare_action(raw_action, human_id, snapshot)
            command = str(action["command"])
            state.active_action = action
            state.action_applied = False
            state.active_command = command
            state.active_target = action.get("target") if command == "goto" else None
            state.last_action = action
            scheduled[human_id] = action
            self._record_scheduled_action(human_id, action, snapshot)
        return scheduled

    def _prepare_action(
        self,
        raw_action: Dict[str, Any],
        human_id: str,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = {
            key: value
            for key, value in raw_action.items()
            if key not in {"human", "after_completion", "until"}
        }
        command = str(action["command"])
        action["command"] = command
        if command == "goto":
            action["target"] = self._resolve_point(action.get("target"))
        if "leave_target" in action:
            action["leave_target"] = self._resolve_point(action["leave_target"])
        if "target_anchor" in action:
            action["target_anchor"] = action["target_anchor"]
        if "seat_anchor" in action:
            seat_prim = self._resolve_seat_prim(action["seat_anchor"])
            if seat_prim:
                action["seat_prim"] = seat_prim
        return action

    def _record_scheduled_action(
        self,
        human_id: str,
        action: Dict[str, Any],
        snapshot: Dict[str, Any],
    ):
        command = str(action.get("command") or "")
        message = str(action.get("message") or "").strip()
        if command != "idle" or message:
            self._turns += 1
        self._replay_log.append(
            {
                **self._event_clock(snapshot),
                "event": "human_action_scheduled",
                "human_id": human_id,
                "action": dict(action),
            }
        )

    def _publish_task_updates(
        self,
        *,
        milestone_id: str,
        source_human_id: str,
        step: Any,
        physics_step: Optional[int] = None,
    ) -> None:
        """Apply typed task updates when their source utterance becomes public."""

        if not milestone_id:
            return
        milestone = next(
            (
                item
                for item in self._milestones()
                if str(item.milestone_id) == milestone_id
            ),
            None,
        )
        if milestone is None:
            raise RuntimeError(
                f"Human action references unknown milestone {milestone_id!r}."
            )
        updates = tuple(milestone.event.task_updates)
        if not updates:
            return
        selected = [
            update
            for update in updates
            if update.source_human_id == source_human_id
            and (milestone_id, update.field) not in self._published_task_updates
        ]
        replacements: dict[str, dict[str, Any]] = {}
        for update in selected:
            previous = self._public_task_values.get(update.field)
            if previous != update.previous_value_ref:
                raise RuntimeError(
                    f"{milestone_id}: expected {update.field}={update.previous_value_ref!r}, "
                    f"found {previous!r}"
                )
            replacements.update(update.outcome_updates)
        if replacements:
            if self.task_bindings is None:
                raise RuntimeError(
                    "Task revisions require an evaluation binding contract"
                )
            self.task_bindings.activate(
                replacements, completed=self._completed_milestone_ids
            )
        for update in updates:
            expected_source = str(update.source_human_id)
            if expected_source != source_human_id:
                continue
            field = str(update.field)
            update_key = (milestone_id, field)
            if update_key in self._published_task_updates:
                continue
            previous = self._public_task_values.get(field)
            expected_previous = update.previous_value_ref
            if previous != expected_previous:
                raise RuntimeError(
                    f"Milestone {milestone_id!r} expected public task value "
                    f"{field}={expected_previous!r}, found {previous!r}."
                )
            self._public_task_values[field] = str(update.new_value_ref)
            self._published_task_updates.add(update_key)
            self._replay_log.append(
                {
                    "step": step,
                    "physics_step": physics_step,
                    "event": "task_value_activated",
                    "milestone_id": milestone_id,
                    "field": field,
                    "previous_value_ref": expected_previous,
                    "new_value_ref": str(update.new_value_ref),
                    "source_human_id": source_human_id,
                    "delivery": str(update.delivery),
                }
            )

    def _resolve_point(self, value: Any) -> List[float]:
        return resolve_human_action_point(self.episode, value)

    def _resolve_seat_prim(self, seat_anchor: Any) -> str:
        target = _normalize_key(seat_anchor)
        anchors = self._world.get("anchors")
        if not isinstance(anchors, dict):
            return ""
        for key, value in anchors.items():
            if _normalize_key(key) != target or not isinstance(value, dict):
                continue
            return str(value.get("prim_path_hint") or "").strip()
        return ""

    def _validate_episode_contract(self):
        question_routes: dict[tuple[tuple[str, ...], str, str], str] = {}
        milestones = self._milestones()
        for milestone in milestones:
            milestone_id = str(milestone.milestone_id)
            trigger = self._milestone_trigger(milestone)
            self._validate_trigger(trigger, milestone_id)
            leaves = self._trigger_leaves(trigger)
            if self._trigger_contains_any(trigger) and any(
                _normalize_key(item.get("type") or "") == "agent question"
                for item in leaves
            ):
                raise ValueError(
                    f"Milestone {milestone_id!r} places an Agent question "
                    "inside a disjunctive trigger; NPC responses require one "
                    "deterministic causal route."
                )
            dependencies = tuple(
                sorted(
                    str(item.milestone_ref)
                    for item in milestone.preconditions
                    if isinstance(item, MilestoneReference)
                )
            )
            for item in leaves:
                if _normalize_key(item.get("type") or "") != "agent question":
                    continue
                key = (
                    dependencies,
                    self._resolve_human_id(item.get("human") or ""),
                    str(item.get("intent_id") or "").strip(),
                )
                previous = question_routes.get(key)
                if previous is not None:
                    raise ValueError(
                        "NPC question responses are ambiguous in the same "
                        f"causal stage: milestones {previous!r} and "
                        f"{milestone_id!r} both bind recipient={key[1]!r}, "
                        f"intent_id={key[2]!r}."
                    )
                question_routes[key] = milestone_id
            for index, action in enumerate(milestone.event.human_actions):
                if not isinstance(action, dict):
                    raise TypeError(
                        f"Milestone {milestone_id} human action #{index} must be a mapping: "
                        f"{action!r}"
                    )
                raw_human_id = action.get("human") or action.get("actor")
                if raw_human_id in (None, ""):
                    raise ValueError(
                        f"Milestone {milestone_id} human action #{index} is missing "
                        f"a human/actor id: {action!r}"
                    )
                human_id = self._resolve_human_id(raw_human_id)
                if human_id not in self._humans:
                    raise ValueError(
                        f"Milestone {milestone_id} human action #{index} references "
                        f"unknown human {raw_human_id!r} (resolved {human_id!r})."
                    )
                if "command" not in action:
                    raise ValueError(
                        f"Milestone {milestone_id} human action #{index} is "
                        f"missing command: {action!r}"
                    )
                if action.get("llm") or str(
                    action.get("message", "")
                ).strip().lower() in {"llm", "<llm>", "$llm"}:
                    raise ValueError(
                        "NPC physical actions and scheduled speech must be authored; LLM action markers are retired"
                    )
                try:
                    HUMAN_RUNTIME_CAPABILITIES.validate_action(action)
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Milestone {milestone_id} human action #{index} is "
                        f"invalid: {action!r}"
                    ) from exc

    @staticmethod
    def _trigger_contains_any(trigger: Any) -> bool:
        if not isinstance(trigger, dict):
            return False
        if "any" in trigger:
            return True
        return any(
            HumanActorRuntime._trigger_contains_any(item)
            for item in trigger.get("all", [])
        )

    def _validate_trigger(self, trigger: Any, milestone_id: str):
        if trigger is None:
            raise ValueError(f"Milestone {milestone_id} is missing a trigger.")
        if isinstance(trigger, str):
            trigger_type = _normalize_key(trigger)
            trigger_data: Dict[str, Any] = {}
        elif isinstance(trigger, dict):
            if "all" in trigger or "any" in trigger:
                key = "all" if "all" in trigger else "any"
                items = trigger.get(key)
                if not isinstance(items, list) or not items:
                    raise ValueError(
                        f"Milestone {milestone_id} trigger.{key} must be a non-empty list."
                    )
                for item in items:
                    self._validate_trigger(item, milestone_id)
                return
            trigger_type = _normalize_key(
                trigger.get("type") or trigger.get("when") or ""
            )
            trigger_data = trigger
        else:
            raise TypeError(
                f"Milestone {milestone_id} trigger must be a string or mapping: {trigger!r}"
            )

        if trigger_type not in SUPPORTED_TRIGGER_TYPES:
            raise ValueError(
                f"Milestone {milestone_id} has unsupported trigger type {trigger_type!r}."
            )
        if trigger_type in {
            "agent question",
            "agent action",
            "robot near human",
        }:
            raw_human_id = trigger_data.get("human")
            if raw_human_id in (None, ""):
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} requires human."
                )
            human_id = self._resolve_human_id(raw_human_id)
            if human_id not in self._humans:
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} references "
                    f"unknown human {raw_human_id!r} (resolved {human_id!r})."
                )
        if trigger_type == "agent question":
            intent_id = str(trigger_data.get("intent_id") or "").strip()
            if not intent_id:
                raise ValueError(
                    f"Milestone {milestone_id} trigger "
                    "'agent question' requires intent_id."
                )
        if trigger_type == "agent action":
            action = str(trigger_data.get("action") or "").strip()
            if not action:
                raise ValueError(
                    f"Milestone {milestone_id} trigger 'agent action' requires action."
                )
        if trigger_type == "robot near position":
            if set(trigger_data) != {"type", "position", "distance_m"}:
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} "
                    "accepts exactly type, position, and distance_m."
                )
            position = trigger_data.get("position")
            if not _is_point(position) or len(position) != 2:
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} "
                    "requires position [x, y]."
                )
            try:
                distance_m = float(trigger_data["distance_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} "
                    "requires numeric distance_m."
                ) from exc
            if not math.isfinite(distance_m) or distance_m <= 0.0:
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} "
                    "requires finite positive distance_m."
                )
        if trigger_type == "robot near human":
            try:
                distance_m = float(trigger_data["distance_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} "
                    "requires numeric distance_m."
                ) from exc
            if distance_m != HUMAN_COMMUNICATION_RADIUS_M:
                raise ValueError(
                    f"Milestone {milestone_id} trigger {trigger_type!r} "
                    f"must use the shared {HUMAN_COMMUNICATION_RADIUS_M:g} m "
                    "communication radius."
                )
        if trigger_type == "milestone condition":
            condition_id = str(trigger_data.get("condition_id") or "").strip()
            if not condition_id:
                raise ValueError(
                    f"Milestone {milestone_id} condition trigger requires condition_id."
                )

    @staticmethod
    def _distance_xy(first: Iterable[Any], second: Iterable[Any]) -> float:
        a = list(first)
        b = list(second)
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
