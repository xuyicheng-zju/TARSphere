"""Runner-facing robot policy with serial Action-Horizon execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tarsphere.harness.core.types import (
    AgentObservation,
    CameraViewObservation,
    EVENT_WINDOW_OUTPUT_FIELDS,
    ToolCall,
    TurnResult,
)
from tarsphere.harness.harness import AgentHarnessConfig
from tarsphere.contracts import DecisionRequirements


ACTION_SPACE_FAMILIES = (
    "cognitive",
    "perceptual",
    "communicative",
    "physical",
)


def declared_episode_actions(
    episode: Any,
    agent_harness_config: AgentHarnessConfig | None = None,
) -> set[str]:
    """Return the run surface implied by an episode's evaluation role."""

    action_spec = getattr(episode, "agent_action_space", None)
    if getattr(episode, "evaluation", None) is not None:
        from tarsphere.bench.compilation.episode import agent_action_space

        action_spec = agent_action_space()
    actions = {
        str(action)
        for family in ACTION_SPACE_FAMILIES
        for action in (
            getattr(action_spec, family, ()) if action_spec is not None else ()
        )
    }
    return actions


def project_agent_feedback(
    feedback: dict[str, Any],
) -> dict[str, Any]:
    """Copy measured execution feedback into the Agent boundary.

    The runner and evaluator retain their original feedback. This projection is
    the single entry to both current feedback and the Agent's text history.
    """
    public = dict(feedback)
    return public


def agent_observation_from_runtime(
    raw: dict[str, Any],
    *,
    camera_views: Any,
) -> AgentObservation:
    """Validate one runtime observation against the requested camera views."""

    raw_views = raw.get("views")
    if not isinstance(raw_views, dict):
        raise ValueError(
            "agent observation requires a 'views' mapping; flat rgb/depth "
            "camera fields are not part of the current protocol"
        )
    views: dict[str, CameraViewObservation] = {}
    for camera in camera_views:
        payload = raw_views.get(camera.value)
        if not isinstance(payload, dict):
            raise ValueError(
                f"observation is missing requested {camera.value!r} camera view"
            )
        if payload.get("rgb") is None:
            raise ValueError(f"{camera.value!r} camera view has no RGB frame")
        views[camera.value] = CameraViewObservation(
            name=camera.value,
            rgb=payload["rgb"],
            depth=payload.get("depth"),
            camera_params=dict(payload.get("camera_params") or {}),
        )
    raw_visible_humans = raw.get("visible_humans") or {}
    if not isinstance(raw_visible_humans, dict):
        raise TypeError("visible_humans must be a mapping of Human ids to XY")
    visible_humans: dict[str, tuple[float, float]] = {}
    for raw_human_id, raw_xy in raw_visible_humans.items():
        human_id = str(raw_human_id).strip()
        if not human_id:
            raise ValueError("visible_humans contains an empty Human id")
        if not isinstance(raw_xy, (list, tuple)) or len(raw_xy) != 2:
            raise ValueError(
                f"visible_humans[{human_id!r}] must contain exactly world XY"
            )
        xy = (float(raw_xy[0]), float(raw_xy[1]))
        if not all(math.isfinite(value) for value in xy):
            raise ValueError(
                f"visible_humans[{human_id!r}] must contain finite world XY"
            )
        visible_humans[human_id] = xy

    return AgentObservation(
        actor_id=str(raw.get("actor_id") or "robot"),
        views=views,
        pose=raw.get("pose"),
        text=str(raw.get("text") or ""),
        public_events=list(raw.get("public_events") or []),
        visible_humans=visible_humans,
        raw=raw,
    )


class RobotAgent:
    """Perceive every action step and invoke the VLM only when needed.

    Every ordinary policy inference proposes the configured Action Horizon of
    executable tool calls, or selects a block of one through five calls when
    the dynamic horizon is enabled. The block is a serial program: each queued
    call is resolved only after the preceding call has returned through the
    runner, using that action step's fresh observation and timestamp. No model
    inference occurs between calls in the same block.
    """

    def __init__(
        self,
        *,
        agent_harness_config: AgentHarnessConfig,
        perception: Any,
        memory: Any,
        kernel: Any,
        tools: Any,
    ) -> None:
        self.agent_harness_config = agent_harness_config
        self.perception = perception
        self.memory = memory
        self.kernel = kernel
        self.tools = tools
        self._action_step = 0
        self._policy_step = 0
        self._pending_actions: list[dict[str, Any]] = []
        self._pending_tool_calls: list[tuple[int, ToolCall]] = []
        self._queued_policy_step: int | None = None
        self._queued_sequence_length = 0
        self._latest_feedback: dict[str, Any] = {}
        self._pending_decision_requirements: DecisionRequirements | None = None
        self._active_memory_step: int | None = None
        self._discarded_actions = 0
        self._deferred_discarded_action_details: list[dict[str, Any]] = []
        self._next_replan_reason: str | None = None
        self._replan_counts: dict[str, int] = {}

    @property
    def execution_stats(self) -> dict[str, Any]:
        """Current policy/action accounting for trace-level efficiency study."""

        return {
            "action_steps": self._action_step,
            "policy_steps": self._policy_step,
            "pending_actions": (
                len(self._pending_actions) + len(self._pending_tool_calls)
            ),
            "pending_tool_calls": len(self._pending_tool_calls),
            "discarded_actions": self._discarded_actions,
            "replan_counts": dict(self._replan_counts),
        }

    def configure_model_input_capture(self, directory: Any) -> None:
        """Bind complete provider-input evidence to this physical attempt."""

        configure = getattr(self.kernel.vlm, "configure_request_capture", None)
        if callable(configure):
            configure(directory)
        elif directory is not None:
            raise RuntimeError("Model client cannot capture formal multimodal inputs")

    def reset(self) -> None:
        self._action_step = 0
        self._policy_step = 0
        self._pending_actions.clear()
        self._pending_tool_calls.clear()
        self._queued_policy_step = None
        self._queued_sequence_length = 0
        self._latest_feedback = {}
        self._pending_decision_requirements = None
        self._active_memory_step = None
        self._discarded_actions = 0
        self._deferred_discarded_action_details.clear()
        self._next_replan_reason = None
        self._replan_counts.clear()
        for component in (self.perception, self.memory, self.kernel):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()

    def discard_pending_actions(
        self,
        reason: str,
        *,
        preserve_decision_requirements: bool = False,
    ) -> list[dict[str, Any]]:
        """Record and drop the queued suffix."""
        if not self._pending_actions and not self._pending_tool_calls:
            if not preserve_decision_requirements:
                self._pending_decision_requirements = None
            return []
        details = [
            self._discarded_action_summary(item, reason)
            for item in self._pending_actions
        ]
        details.extend(
            self._discarded_tool_call_summary(index, call, reason)
            for index, call in self._pending_tool_calls
        )
        self._discarded_actions += len(details)
        self._pending_actions.clear()
        self._pending_tool_calls.clear()
        self._queued_policy_step = None
        self._queued_sequence_length = 0
        if not preserve_decision_requirements:
            self._pending_decision_requirements = None
        return details

    @staticmethod
    def _event_requires_replan(
        requirements: DecisionRequirements | None,
    ) -> bool:
        """Return whether a newly delivered event requires a policy response."""

        return bool(requirements and requirements.event_milestone_ids)

    @staticmethod
    def _external_public_event_delivered(observation: Any) -> bool:
        """Return whether this observation delivers an event from outside the robot."""

        if isinstance(observation, Mapping):
            actor_id = str(observation.get("actor_id") or "robot")
            events = observation.get("public_events") or ()
        else:
            actor_id = str(getattr(observation, "actor_id", "robot") or "robot")
            events = getattr(observation, "public_events", None) or ()
        actor_key = actor_id.strip().casefold()
        for event in events:
            if not isinstance(event, Mapping):
                return True
            speaker = str(event.get("speaker") or "").strip().casefold()
            if not speaker or speaker not in {actor_key, "robot"}:
                return True
        return False

    @classmethod
    def _decision_boundary_requires_replan(
        cls,
        *,
        observation: Any,
        requirements: DecisionRequirements | None,
    ) -> bool:
        return cls._event_requires_replan(
            requirements
        ) or cls._external_public_event_delivered(observation)

    def requires_policy_inference(
        self,
        decision_requirements: DecisionRequirements | None = None,
        *,
        observation: Any = None,
    ) -> bool:
        """Report whether the next action needs a new model decision.

        The runner uses this before admitting an action at the policy budget.
        This check does not mutate the queue or consume event requirements.
        """

        if self._decision_boundary_requires_replan(
            observation=observation,
            requirements=decision_requirements,
        ):
            return True
        return not bool(self._pending_actions or self._pending_tool_calls)

    @staticmethod
    def _feedback_requires_replan(feedback: dict[str, Any]) -> bool:
        """Return whether ordinary feedback invalidates a queued suffix."""

        if feedback.get("safety_violation") or feedback.get("dropped_objects"):
            return False
        navigation_stuck = feedback.get("navigation_stuck")
        if isinstance(navigation_stuck, dict) and navigation_stuck.get("terminal"):
            return False
        return feedback.get("executed") is False

    def _defer_discarded_actions(
        self,
        details: list[dict[str, Any]],
        *,
        replan_reason: str,
    ) -> None:
        if not details:
            return
        self._deferred_discarded_action_details.extend(details)
        self._next_replan_reason = replan_reason

    @staticmethod
    def _merge_decision_requirements(
        existing: DecisionRequirements | None,
        incoming: DecisionRequirements | None,
    ) -> DecisionRequirements | None:
        """Retain event requirements until the next model inference.

        A public event may occur while a model-authored serial block is still
        executing. It must not interrupt that block, and its required output
        must not disappear before the next policy decision.
        """

        if existing is None:
            return incoming
        if incoming is None:
            return existing
        return DecisionRequirements(
            planning=(
                {}
                if existing.planning is not None or incoming.planning is not None
                else None
            ),
            event_milestone_ids=tuple(
                dict.fromkeys(
                    (
                        *existing.event_milestone_ids,
                        *incoming.event_milestone_ids,
                    )
                )
            ),
        )

    @staticmethod
    def _serialized_tool_results(results: list[Any]) -> list[dict[str, Any]]:
        """Keep only model-visible/auditable fields from handler results."""

        return [
            {
                "name": result.name,
                "content": result.content,
                "is_error": bool(result.is_error),
                **(
                    {"error_type": str(result.metadata["error_type"])}
                    if (
                        result.is_error
                        and isinstance(result.metadata, dict)
                        and result.metadata.get("error_type")
                    )
                    else {}
                ),
            }
            for result in results
        ]

    @staticmethod
    def _evaluator_perception(results: list[Any]) -> list[dict[str, Any]]:
        """Extract evaluator-only localization measurements."""

        return [
            {
                "tool": "locate_object",
                "target": str(result.metadata.get("target") or ""),
                "semantic_score": float(result.metadata["score"]),
                "target_center_world_m": [
                    float(value) for value in result.metadata["target_center_world_m"]
                ],
            }
            for result in results
            if (
                result.name == "locate_object"
                and not result.is_error
                and isinstance(result.metadata, dict)
                and isinstance(
                    result.metadata.get("target_center_world_m"),
                    (list, tuple),
                )
                and len(result.metadata["target_center_world_m"]) == 3
                and result.metadata.get("score") is not None
            )
        ]

    @staticmethod
    def _discarded_tool_call_summary(
        index: int,
        call: ToolCall,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "call_name": call.tool,
            "action": call.tool,
            "action_index": int(index),
            "tool_call_index": int(index),
            "reason": str(reason),
        }

    def configure_episode_humans(self, humans: list[Any]) -> None:
        """Bind the episode cast to episode-long public Human memory."""

        assets = [str(getattr(human, "asset", "") or "").strip() for human in humans]
        assets = [asset for asset in assets if asset]
        from tarsphere.humans.character_catalog import (
            public_character_profiles_for_assets,
        )

        profiles = public_character_profiles_for_assets(assets)
        for human in humans:
            if not str(getattr(human, "scene_person_prim_path", "") or "").strip():
                continue
            profiles.append(
                {
                    "name": str(human.id),
                    "caption": str(human.public_description),
                    "job": str(human.public_job),
                }
            )
        profile_names = tuple(
            sorted(str(profile["name"]).strip() for profile in profiles)
        )
        declared_names = tuple(
            sorted(
                {
                    str(getattr(human, "name", "") or "").strip()
                    for human in humans
                    if str(getattr(human, "name", "") or "").strip()
                }
            )
        )
        if declared_names != profile_names:
            raise ValueError(
                "Episode Human names must match their public character "
                f"profiles: declared={declared_names}, catalog={profile_names}."
            )
        by_name = {str(human.id): human for human in humans}
        for profile in profiles:
            human = by_name[profile["name"]]
            if getattr(human, "asset", None):
                # The catalog owns appearance, not the role in this episode.
                job = str(getattr(human, "public_job", "") or "").strip()
                description = str(
                    getattr(human, "public_description", "") or ""
                ).strip()
                if job:
                    profile["job"] = job
                if description:
                    profile["caption"] = (
                        f"{profile['caption']} Episode context: {description}"
                    )
            facts = getattr(human, "public_profile_facts", ())
            if facts:
                profile["public_profile_facts"] = [fact.model_dump() for fact in facts]
        self.memory.configure_human_profiles(profiles)

    def configure_episode(self, episode: Any) -> None:
        """Bind the episode contract before the first observation."""

        configure_kernel = getattr(self.kernel, "configure_episode", None)
        if callable(configure_kernel):
            configure_kernel(episode)

        declared_actions = declared_episode_actions(
            episode,
            self.agent_harness_config,
        )
        registered_actions = set(self.tools.spec_names())
        missing_actions = sorted(declared_actions - registered_actions)
        if missing_actions:
            raise RuntimeError(
                "The run requires Agent tools whose backends are not wired: "
                f"{missing_actions}. Available tools are "
                f"{sorted(registered_actions)}."
            )

    def update_episode_progress(self, milestone_ids: Any) -> None:
        """Update private causal gates without exposing evaluator state."""

        update = getattr(self.kernel, "update_episode_progress", None)
        if callable(update):
            update(milestone_ids)

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        action_space: list[str] | None = None,
        *,
        decision_requirements: Any = None,
    ) -> dict[str, Any]:
        """Return the next call in the model-authored serial block."""

        agent_obs = self._to_agent_observation(observation)
        perception_output = self.perception.process(
            agent_obs,
            metadata={
                "action_step": self._action_step,
                "visual_modality": "rgb",
            },
        )
        memory_state = self.memory.begin_action_step(
            observation=agent_obs,
            perception=perception_output,
            action_step=self._action_step,
            record_visual_history=True,
        )
        self._active_memory_step = self._action_step

        incoming_event_replan = self._decision_boundary_requires_replan(
            observation=agent_obs,
            requirements=decision_requirements,
        )
        self._pending_decision_requirements = self._merge_decision_requirements(
            self._pending_decision_requirements,
            decision_requirements,
        )
        if incoming_event_replan and (
            self._pending_actions or self._pending_tool_calls
        ):
            discarded = self.discard_pending_actions(
                "public_event_replan",
                preserve_decision_requirements=True,
            )
            self._defer_discarded_actions(
                discarded,
                replan_reason="public_event",
            )

        has_pending = bool(self._pending_actions or self._pending_tool_calls)
        policy_inference = not has_pending
        replan_reason: str | None = None
        if policy_inference:
            replan_reason = self._next_replan_reason or (
                "initial_policy"
                if self._policy_step == 0
                else "action_horizon_exhausted"
            )
            self._next_replan_reason = None
            self._replan_counts[replan_reason] = (
                self._replan_counts.get(replan_reason, 0) + 1
            )

        observe_private_mapping = getattr(
            self.kernel,
            "observe_private_mapping",
            None,
        )
        if callable(observe_private_mapping):
            observe_private_mapping(
                observation=agent_obs,
                action_space=action_space,
                action_step=self._action_step,
                update_frontiers=policy_inference,
                saved_entities=self.memory.semantic.landmarks,
            )

        if policy_inference:
            model_visible_action_steps = sorted(
                {int(frame.action_step) for frame in memory_state.visual_history}
            )
            turn_result: TurnResult = self.kernel.run(
                observation=agent_obs,
                instruction=instruction,
                memory=memory_state,
                action_space=action_space,
                perception=perception_output,
                feedback=self._latest_feedback,
                decision_requirements=self._pending_decision_requirements,
                policy_step=self._policy_step,
                action_step=self._action_step,
            )
            if not turn_result.actions:
                if turn_result.status in {"rejected", "error"}:
                    # One policy inference always produces one public outcome.
                    # Semantic/format rejection is non-terminal and becomes
                    # the next decision's feedback. Only a provider request
                    # that exhausted VLMClient's network recovery is terminal.
                    policy_step = self._policy_step
                    self._policy_step += 1
                    self._queued_policy_step = None
                    self._queued_sequence_length = 0
                    terminal_model_failure = turn_result.status == "error"
                    if terminal_model_failure:
                        self._pending_decision_requirements = None
                    envelope = turn_result.envelope
                    serialized_envelope = (
                        envelope.as_dict()
                        if envelope is not None
                        else {"thinking": "", "actions": []}
                    )
                    diagnostic = turn_result.diagnostic or "policy output rejected"
                    if not terminal_model_failure:
                        error_type = "policy_output_rejected"
                    elif turn_result.failure_kind == ("transport_retry_exhausted"):
                        error_type = "model_service_retry_exhausted"
                    else:
                        error_type = "model_request_failed"
                    action = {
                        "action": "policy_rejected",
                        "call_name": "policy_rejected",
                        "tool_args": {},
                        "policy_step": policy_step,
                        "action_step": self._action_step,
                        "action_index": 0,
                        "tool_call_index": 0,
                        "action_sequence_length": 0,
                        "envelope": serialized_envelope,
                        "accepted_event_output_batches": [],
                        "tool_results": [
                            {
                                "name": "policy_rejected",
                                "content": diagnostic,
                                "is_error": True,
                                "error_type": error_type,
                            }
                        ],
                        "evaluator_perception": [],
                        "model_calls": [dict(call) for call in turn_result.model_calls],
                        "model_visible_action_steps": model_visible_action_steps,
                        "policy_inference": True,
                        "replan_reason": replan_reason,
                        "is_error": True,
                        "policy_status": turn_result.status,
                        "policy_diagnostic": diagnostic,
                        "terminal_model_failure": terminal_model_failure,
                        "terminal_model_failure_reason": (
                            error_type if terminal_model_failure else ""
                        ),
                        "discarded_actions": 0,
                        "discarded_action_details": [],
                        "agent_harness_config": self.agent_harness_config.as_dict(),
                    }
                    if self._deferred_discarded_action_details:
                        deferred = list(self._deferred_discarded_action_details)
                        action["discarded_actions"] = len(deferred)
                        action["discarded_action_details"] = deferred
                        self._deferred_discarded_action_details.clear()
                    self.memory.record_executed_action(
                        policy_step=policy_step,
                        action=action,
                    )
                    self._action_step += 1
                    return action
                error = RuntimeError(
                    "Robot policy turn ended without an executable sequence: "
                    f"status={turn_result.status!r}, "
                    f"diagnostic={turn_result.diagnostic!r}"
                )
                # The runner has no trace row yet, but these provider calls
                # still happened and must remain billable/auditable.
                error.model_calls = [dict(call) for call in turn_result.model_calls]
                raise error
            self._pending_decision_requirements = None
            envelope = turn_result.envelope
            if envelope is None:
                raise RuntimeError("completed policy turn has no DecisionEnvelope")
            policy_step = self._policy_step
            sequence_length = len(envelope.actions)
            self._queued_policy_step = policy_step
            self._queued_sequence_length = sequence_length
            self._pending_tool_calls = (
                [
                    (index, call)
                    for index, call in enumerate(turn_result.pending_calls, start=1)
                ]
                if getattr(turn_result, "pending_calls", None)
                else []
            )
            discarded_calls = list(getattr(turn_result, "discarded_calls", None) or [])
            executable_count = 1 + len(self._pending_tool_calls)
            discarded_details = [
                self._discarded_tool_call_summary(
                    executable_count + index,
                    call,
                    "feedback_boundary",
                )
                for index, call in enumerate(discarded_calls)
            ]
            self._discarded_actions += len(discarded_details)

            serialized_envelope = envelope.as_dict()
            serialized_results = self._serialized_tool_results(turn_result.tool_results)
            evaluator_perception = self._evaluator_perception(turn_result.tool_results)
            self._pending_actions = []
            for runtime_index, payload in enumerate(turn_result.actions):
                item = dict(payload)
                item["policy_step"] = policy_step
                item["action_index"] = int(item.get("tool_call_index", 0))
                item["action_sequence_length"] = sequence_length
                item["envelope"] = (
                    serialized_envelope
                    if runtime_index == 0
                    else {
                        key: value
                        for key, value in serialized_envelope.items()
                        if key not in EVENT_WINDOW_OUTPUT_FIELDS
                    }
                )
                if runtime_index == 0:
                    item["accepted_event_output_batches"] = (
                        turn_result.accepted_event_output_batches
                    )
                    item["discarded_actions"] = len(discarded_details)
                    item["discarded_action_details"] = discarded_details
                item["tool_results"] = serialized_results
                item["evaluator_perception"] = evaluator_perception
                item["model_calls"] = (
                    [dict(call) for call in turn_result.model_calls]
                    if runtime_index == 0
                    else []
                )
                item["model_visible_action_steps"] = (
                    model_visible_action_steps if runtime_index == 0 else []
                )
                item["policy_inference"] = runtime_index == 0
                item["replan_reason"] = replan_reason if runtime_index == 0 else None
                self._pending_actions.append(item)

            proposed_actions = [
                {
                    "action": call.tool,
                    "call_name": call.tool,
                    "tool_call_index": index,
                    "args": dict(call.args),
                }
                for index, call in enumerate(envelope.actions)
            ]
            self.memory.record_policy_decision(
                policy_step=policy_step,
                envelope=envelope,
                actions=proposed_actions,
                accepted_memory_update_batches=(
                    turn_result.accepted_memory_update_batches
                ),
                plan_updates_accepted=turn_result.plan_updates_accepted,
                accepted_event_output_batches=turn_result.accepted_event_output_batches,
            )
            self._policy_step += 1
        elif not self._pending_actions:
            if not self._pending_tool_calls:
                raise RuntimeError("serial action block lost its queued suffix")
            tool_index, call = self._pending_tool_calls.pop(0)
            if self._queued_policy_step is None:
                raise RuntimeError("queued tool call has no owning policy step")
            policy_step = int(self._queued_policy_step)
            turn_result = self.kernel.resolve_queued_call(
                call=call,
                observation=agent_obs,
                instruction=instruction,
                memory=memory_state,
                action_space=action_space,
                perception=perception_output,
                feedback=self._latest_feedback,
                action_step=self._action_step,
            )
            if not turn_result.actions:
                raise RuntimeError(
                    "Queued tool call ended without a runtime payload: "
                    f"{call.tool}: {turn_result.diagnostic}"
                )
            call_envelope = {
                "actions": [
                    {
                        "tool": call.tool,
                        "args": dict(call.args),
                    }
                ],
            }
            serialized_results = self._serialized_tool_results(turn_result.tool_results)
            evaluator_perception = self._evaluator_perception(turn_result.tool_results)
            self._pending_actions = []
            for payload in turn_result.actions:
                item = dict(payload)
                item["policy_step"] = policy_step
                item["action_index"] = int(tool_index)
                item["tool_call_index"] = int(tool_index)
                item["action_sequence_length"] = int(self._queued_sequence_length)
                item["envelope"] = call_envelope
                item["tool_results"] = serialized_results
                item["evaluator_perception"] = evaluator_perception
                item["model_calls"] = []
                item["model_visible_action_steps"] = []
                item["policy_inference"] = False
                item["replan_reason"] = None
                self._pending_actions.append(item)

        if not self._pending_actions:
            raise RuntimeError("serial action block produced no executable action")
        action = dict(self._pending_actions.pop(0))
        policy_step = int(action["policy_step"])

        action["action_step"] = self._action_step
        action.setdefault("policy_inference", False)
        action.setdefault("model_visible_action_steps", [])
        action.setdefault("replan_reason", None)
        action.setdefault("discarded_actions", 0)
        action.setdefault("discarded_action_details", [])
        if self._deferred_discarded_action_details:
            prior = list(action["discarded_action_details"])
            deferred = list(self._deferred_discarded_action_details)
            action["discarded_action_details"] = [*prior, *deferred]
            action["discarded_actions"] = len(action["discarded_action_details"])
            self._deferred_discarded_action_details.clear()
        action["agent_harness_config"] = self.agent_harness_config.as_dict()
        action["call_name"] = action.get("call_name")
        action["tool_call_index"] = int(
            action.get("tool_call_index", action.get("action_index", 0))
        )
        action["action_index"] = int(action.get("action_index", 0))
        action["action_sequence_length"] = int(action.get("action_sequence_length", 1))

        self.memory.record_executed_action(
            policy_step=policy_step,
            action=action,
        )
        self._action_step += 1
        if not self._pending_actions and not self._pending_tool_calls:
            self._queued_policy_step = None
            self._queued_sequence_length = 0
        return action

    @staticmethod
    def _discarded_action_summary(
        action: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """Persist a resolved call skipped by an episode terminal."""

        return {
            "call_name": str(action.get("call_name") or ""),
            "action": str(action.get("action") or ""),
            "action_index": int(action.get("action_index", 0) or 0),
            "tool_call_index": int(
                action.get("tool_call_index", action.get("action_index", 0)) or 0
            ),
            "reason": str(reason),
        }

    def observe_private_mapping_frame(
        self,
        *,
        observation: dict[str, Any],
        action_space: list[str],
        physics_step: int,
        frame_index: int,
    ) -> dict[str, Any]:
        """Fuse one policy-invisible motion-time RGB-D frame.

        The fixed-rate mapping path deliberately bypasses perception,
        episodic/visual memory, planning, and policy inference. It can only
        mutate the private online occupancy grid.
        """

        observe_mapping = getattr(
            self.kernel,
            "observe_private_mapping",
            None,
        )
        if not callable(observe_mapping):
            raise RuntimeError(
                "The configured Agent kernel has no private mapping sink."
            )
        executing_action_step = max(0, int(self._action_step) - 1)
        return observe_mapping(
            observation=self._to_agent_observation(observation),
            action_space=action_space,
            action_step=executing_action_step,
            update_frontiers=False,
            artifact_stem=(
                f"online_map_motion_frame_{int(frame_index):06d}_"
                f"action_{executing_action_step:04d}_"
                f"physics_{int(physics_step):08d}"
            ),
        )

    def update_info(self, info: dict[str, Any]) -> None:
        """Complete the active action-step record with public execution facts."""

        if self._active_memory_step is None:
            raise RuntimeError(
                "received execution feedback without an active action step"
            )
        feedback = project_agent_feedback(
            info,
        )
        feedback.setdefault("action_step", self._active_memory_step)
        self._latest_feedback = feedback
        self.memory.complete_action_step(feedback)
        self._active_memory_step = None
        if self._feedback_requires_replan(feedback) and (
            self._pending_actions or self._pending_tool_calls
        ):
            discarded = self.discard_pending_actions(
                "execution_failure_replan",
                preserve_decision_requirements=True,
            )
            self._defer_discarded_actions(
                discarded,
                replan_reason="execution_failure",
            )
        update_kernel = getattr(self.kernel, "update_info", None)
        if callable(update_kernel):
            update_kernel(feedback)

    def _to_agent_observation(self, raw: dict[str, Any]) -> AgentObservation:
        """Validate the current-only normalized multi-view observation shape."""
        return agent_observation_from_runtime(
            raw,
            camera_views=self.agent_harness_config.protocol.camera_views,
        )


__all__ = [
    "ACTION_SPACE_FAMILIES",
    "RobotAgent",
    "agent_observation_from_runtime",
    "declared_episode_actions",
]
