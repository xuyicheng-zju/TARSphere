"""EnvelopeKernel — the active turn loop for strict JSON decisions.

Pipeline per decision:
1. Build the user message (image + instruction + memory feedback).
2. Build the system prompt = base role preamble + envelope schema section
   from prompt_builder (which injects the eval-scope vocab verbatim, D11).
3. Call the VLM (no function-calling; the model emits one JSON object).
4. Parse the raw content with DecisionParser -> DecisionEnvelope.
5. Resolve the first ``tool_call`` through ToolRegistry and retain the authored
   suffix. Each later call resolves from its own fresh action-step state.
6. If a planning field is present, ingest it into StrategyPlanner.
7. Return TurnResult with the envelope attached.

The kernel is stateless across decisions except for the StrategyPlanner it
holds (active-plan memory). Event-specific structured-output requirements are
passed in per-call via ``run(decision_requirements=...)`` so the agent / runner
controls event-driven gating.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
import time
from typing import Any

from tarsphere.harness.core.types import (
    AgentObservation,
    DecisionEnvelope,
    EVENT_WINDOW_OUTPUT_FIELDS,
    ToolCall,
    ToolResult,
    TurnResult,
)
from tarsphere.harness.kernel.decision_parser import DecisionParseError, parse_decision
from tarsphere.harness.kernel.prompt_builder import (
    envelope_schema_section,
)
from tarsphere.harness.core.failures import (
    ARGUMENT_FAILURES,
    BACKEND_FAILURES,
    FailureReason,
    WORLD_FAILURES,
    tool_failure,
    unexpected_tool_failure,
)
from tarsphere.contracts import DecisionRequirements
from tarsphere.harness.planner.strategy import StrategyPlanner
from tarsphere.harness.memory.key import KeyMemoryLedger, KeyMemoryRecord
from tarsphere.harness.memory.semantic import SemanticEntityRecord
from tarsphere.harness.tools.availability import (
    AvailableTools,
    LOCOMOTION_TOOLS,
    resolve_available_tools,
)
from tarsphere.infra.manipulation.agent_grasp import GRASP_FAILURE_CODES
from tarsphere.harness.tools.registry import ToolRegistry
from tarsphere.harness.harness import (
    ProtocolConfig,
)
from tarsphere.infra.perception.lidar import LidarScan, describe_lidar


BASE_SYSTEM_PREAMBLE = (
    "You are an embodied robot in a shared human environment. Use only the public task, "
    "current images, spatial memory, dialogue, and measured execution evidence. "
    "Never invent evaluator-private targets or future events.\n"
)


# Locomotion feedback is control context, not episode knowledge. TextMemory
# applies the configured history horizon before this formatter runs; rendering
# must not silently impose a second, contradictory limit.
NAVIGATION_DEBUG_ARTIFACTS_ENV = "TARSPHERE_NAVIGATION_DEBUG_ARTIFACTS"


def _navigation_debug_artifacts_enabled() -> bool:
    """Return whether private mapping/planner intermediates should be saved."""

    import os

    raw = os.environ.get(NAVIGATION_DEBUG_ARTIFACTS_ENV, "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{NAVIGATION_DEBUG_ARTIFACTS_ENV} must be a boolean flag, got {raw!r}."
    )


class EnvelopeKernel:
    """Dispatch one strict JSON decision envelope over the tool registry."""

    def __init__(
        self,
        *,
        vlm: Any,
        tools: ToolRegistry,
        protocol: ProtocolConfig,
        strategy_planner: StrategyPlanner | None = None,
        system_preamble: str = BASE_SYSTEM_PREAMBLE,
    ) -> None:
        self.vlm = vlm
        self.tools = tools
        self.protocol = protocol
        self.strategy_planner = strategy_planner or StrategyPlanner()
        self.system_preamble = system_preamble
        # Private occupancy grid for A* PointGoal navigation. It is
        # created lazily, accumulated across decisions, and reset per episode.
        self._online_occupancy_grid: Any | None = None
        self._planning_occupancy_grid: Any | None = None
        self._online_map_action_step: int | None = None
        self._online_map_physics_step: int | None = None
        self._online_map_error: str | None = None
        self._open_frontier_overlay_rgb: Any | None = None
        self._open_frontier_exploration_memory_rgb: Any | None = None
        self._open_frontier_action_step: int | None = None
        self._frozen_available_tools: AvailableTools | None = None
        self._frozen_action_space: frozenset[str] | None = None
        self._public_interaction_intents: tuple[dict[str, Any], ...] = ()
        self._completed_episode_milestones: frozenset[str] = frozenset()
        # Only Human-free component diagnostics receive their episode's
        # explicit task here. Official tasks are disclosed through public events.
        self._component_task: str = ""

    def reset(self) -> None:
        self.strategy_planner.reset()
        self._online_occupancy_grid = None
        self._planning_occupancy_grid = None
        self._online_map_action_step = None
        self._online_map_physics_step = None
        self._online_map_error = None
        self._open_frontier_overlay_rgb = None
        self._open_frontier_exploration_memory_rgb = None
        self._open_frontier_action_step = None
        self._frozen_available_tools = None
        self._frozen_action_space = None
        self._public_interaction_intents = ()
        self._completed_episode_milestones = frozenset()
        self._component_task = ""
        # Candidate marks are capture-scoped and must not cross episode resets.
        for name in ("locate_object", "remember_entity"):
            handler = self.tools.handler(name)
            clear = getattr(handler, "reset_pending", None)
            if callable(clear):
                clear()
        for name in (
            "navigate_to_frontier",
            "navigate_to_pointgoal",
            "navigate_to_entity",
        ):
            navigation = self.tools.handler(name)
            reset_navigation = getattr(navigation, "reset", None)
            if callable(reset_navigation):
                reset_navigation()

    def configure_episode(self, episode: Any) -> None:
        """Bind episode-scoped navigation and public interaction affordances."""

        self._online_occupancy_grid = None
        self._planning_occupancy_grid = None
        self._component_task = (
            episode.system_prompt.strip()
            if episode.evaluation is None and not episode.humans
            else ""
        )
        self._public_interaction_intents = tuple(
            {
                "intent_id": str(intent.intent_id),
                "description": str(intent.description),
                "eligible_human_ids": tuple(intent.eligible_human_ids),
                # Preconditions stay harness-private. They control when the
                # affordance becomes visible but are never rendered into the
                # model prompt themselves.
                "required_milestones": tuple(
                    str(reference.milestone_ref)
                    for reference in getattr(intent, "preconditions", ())
                ),
            }
            for intent in episode.interaction.intents
            if intent.public
        )
        self._completed_episode_milestones = frozenset()
        for name in (
            "navigate_to_frontier",
            "navigate_to_pointgoal",
            "navigate_to_entity",
        ):
            navigation = self.tools.handler(name)
            configure_navigation = getattr(
                navigation,
                "configure_episode",
                None,
            )
            if callable(configure_navigation):
                configure_navigation(episode)

    def update_episode_progress(self, milestone_ids: Any) -> None:
        """Advance the private gate used for public interaction affordances."""

        completed = frozenset(str(item) for item in milestone_ids)
        if not self._completed_episode_milestones.issubset(completed):
            raise ValueError("completed episode milestones cannot move backward")
        self._completed_episode_milestones = completed

    def _available_public_interaction_intents(self) -> list[dict[str, Any]]:
        """Return only public topics whose causal prerequisites are complete."""

        available: list[dict[str, Any]] = []
        for intent in self._public_interaction_intents:
            required = frozenset(intent.get("required_milestones") or ())
            if not required.issubset(self._completed_episode_milestones):
                continue
            available.append(
                {
                    "intent_id": str(intent["intent_id"]),
                    "description": str(intent["description"]),
                    "eligible_human_ids": tuple(intent.get("eligible_human_ids") or ()),
                }
            )
        return available

    def observe_private_mapping(
        self,
        *,
        observation: AgentObservation,
        action_space: list[str] | None,
        action_step: int,
        update_frontiers: bool = True,
        artifact_stem: str | None = None,
        saved_entities: tuple[SemanticEntityRecord, ...] = (),
    ) -> dict[str, Any]:
        """Integrate every synchronized ego RGBD observation into local state.

        Mapping is runtime perception, not a side effect of asking the model to
        navigate. It therefore also runs on queued-action observations and
        private fixed-frequency captures without extra model calls.
        """
        declared = set(action_space or ())
        if not declared.intersection(
            {
                "navigate_to_pointgoal",
                "navigate_to_frontier",
                "navigate_to_entity",
            }
        ):
            return {"integrated": False, "reason": "navigation_not_declared"}
        grid = self._ensure_online_occupancy_grid(observation)
        ego = observation.views.get("ego")
        pose = observation.pose
        if ego is None or ego.depth is None or not isinstance(pose, dict):
            self._online_map_error = (
                "latest observation lacks synchronized ego depth or robot pose"
            )
            self._online_map_action_step = int(action_step)
            return {"integrated": False, "reason": self._online_map_error}
        params = ego.camera_params
        if not isinstance(params, dict):
            self._online_map_error = (
                "latest ego depth lacks calibrated camera parameters"
            )
            self._online_map_action_step = int(action_step)
            return {"integrated": False, "reason": self._online_map_error}
        physics_step = params.get("physics_step")
        physics_step = int(physics_step) if physics_step is not None else None
        duplicate_frame = bool(
            physics_step is not None and physics_step == self._online_map_physics_step
        )
        try:
            import numpy as np
            from tarsphere.geometry import fov_to_intrinsics
            from tarsphere.infra.perception.online_mapping import (
                ONLINE_DEPTH_SUBSAMPLE,
            )

            depth = np.asarray(ego.depth, dtype=np.float32)
            position = np.asarray(pose["position"], dtype=np.float64).ravel()
            world_T_cam = np.asarray(params["world_T_cam"], dtype=np.float64)
            if depth.ndim != 2 or position.size < 2 or world_T_cam.shape != (4, 4):
                raise ValueError(
                    "invalid synchronized depth, robot position, or camera transform shape"
                )
            height, width = depth.shape
            K = fov_to_intrinsics(float(params["fov_deg"]), width, height)
            coverage_before = grid.coverage_stats()
            fusion_started = time.monotonic()
            if not duplicate_frame:
                lidar = (
                    observation.raw.get("lidar")
                    if isinstance(observation.raw, dict)
                    and isinstance(observation.raw.get("lidar"), dict)
                    else None
                )
                grid.update_from_depth_with_K(
                    depth=depth,
                    K=K,
                    world_T_cam=world_T_cam,
                    robot_xy=position[:2],
                    lidar=lidar,
                    robot_yaw_rad=(
                        float(pose["yaw_rad"])
                        if pose.get("yaw_rad") is not None
                        else None
                    ),
                    subsample=ONLINE_DEPTH_SUBSAMPLE,
                )
                self._online_map_physics_step = physics_step
                # Audit the actual fusion argument, not the protocol flag.
                # Both decision-time and private motion frames take this path;
                # this text stays in the worker log, outside Agent feedback.
                print(
                    "[MAP-FUSION] "
                    f"action_step={int(action_step)} physics_step={physics_step} "
                    f"frame={'motion' if artifact_stem is not None else 'action'} "
                    f"map_source={grid.map_source} "
                    f"lidar_input={str(lidar is not None).lower()}",
                    flush=True,
                )
            coverage_after = grid.coverage_stats()
            fusion_seconds = time.monotonic() - fusion_started
        except (KeyError, TypeError, ValueError) as exc:
            self._online_map_error = f"{type(exc).__name__}: {exc}"
            self._online_map_action_step = int(action_step)
            return {"integrated": False, "reason": self._online_map_error}
        self._online_map_error = None
        self._online_map_action_step = int(action_step)

        import os

        record_dir = os.environ.get("TARSPHERE_RECORD_DIR", "")
        debug_artifacts = _navigation_debug_artifacts_enabled()
        artifact_started = time.monotonic()
        if (
            record_dir
            and physics_step is not None
            and not duplicate_frame
            and os.environ.get("TARSPHERE_PRESENTATION_CAPTURE") == "1"
        ):
            from pathlib import Path
            from tarsphere.infra.perception.presentation_memory import (
                capture_voxel_snapshot,
            )
            from tarsphere.infra.perception.presentation_rgbd import (
                capture_rgbd_snapshot,
            )

            capture_voxel_snapshot(grid, Path(record_dir), physics_step)
            capture_rgbd_snapshot(
                grid,
                Path(record_dir),
                physics_step,
                rgb=ego.rgb,
                depth=depth,
                K=K,
                world_T_cam=world_T_cam,
                robot_xy=position[:2],
                robot_yaw_rad=pose["yaw_rad"],
            )
        if record_dir and debug_artifacts:
            stem = artifact_stem or (
                f"online_map_observation_step_{int(action_step):04d}"
            )
            grid.dump_png(
                os.path.join(record_dir, f"{stem}.png"),
                robot_xy=position[:2],
                include_planning_buffer=True,
            )
            grid.dump_npz(os.path.join(record_dir, f"{stem}.npz"))
            if artifact_stem is not None:
                rgbd_png = (
                    observation.raw.get("rgbd_png")
                    if isinstance(observation.raw, dict)
                    else None
                )
                if not isinstance(rgbd_png, bytes):
                    from tarsphere.infra.perception.rgbd_utils import (
                        compose_rgbd_image,
                    )

                    rgbd_png = compose_rgbd_image(ego.rgb, depth)
                with open(os.path.join(record_dir, f"{stem}_rgbd.png"), "wb") as stream:
                    stream.write(rgbd_png)
        artifact_seconds = time.monotonic() - artifact_started

        if update_frontiers:
            self._refresh_frontier_visuals(
                observation=observation,
                action_step=action_step,
                grid=grid,
                saved_entities=saved_entities,
                record_dir=record_dir or None,
                record_debug_artifacts=debug_artifacts,
            )
        return {
            "integrated": not duplicate_frame,
            "reason": "duplicate_physics_step" if duplicate_frame else "integrated",
            "physics_step": physics_step,
            "observed_area_before_m2": float(coverage_before["observed_area_m2"]),
            "observed_area_after_m2": float(coverage_after["observed_area_m2"]),
            "fusion_seconds": float(fusion_seconds),
            "artifact_seconds": float(artifact_seconds),
        }

    def _refresh_frontier_visuals(
        self,
        *,
        observation: AgentObservation,
        action_step: int,
        grid: Any,
        saved_entities: tuple[SemanticEntityRecord, ...],
        record_dir: str | None = None,
        record_debug_artifacts: bool = False,
    ) -> None:
        navigation = self.tools.handler("navigate_to_frontier")
        backend = getattr(navigation, "backend", None)
        if backend is None:
            raise RuntimeError("frontier navigation has no wired backend")
        _, ego_overlay, exploration_memory, _ = backend.process_observation(
            observation=observation,
            action_step=int(action_step),
            occupancy_grid=grid,
            record_dir=record_dir,
            record_debug_artifacts=record_debug_artifacts,
            saved_entities=saved_entities,
        )
        self._open_frontier_overlay_rgb = ego_overlay
        self._open_frontier_exploration_memory_rgb = exploration_memory
        self._open_frontier_action_step = int(action_step)

    def _ensure_online_occupancy_grid(
        self,
        observation: AgentObservation,
    ) -> Any:
        if self._online_occupancy_grid is None:
            import numpy as np
            from tarsphere.infra.perception.online_mapping import (
                OnlineOccupancyGrid,
            )

            resolution_m = 0.05
            half_extent_m = 24.0
            origin = (-half_extent_m, -half_extent_m)
            pose = observation.pose
            if isinstance(pose, dict) and pose.get("position") is not None:
                position = np.asarray(pose["position"], dtype=np.float64).ravel()
                if position.size >= 2 and np.all(np.isfinite(position[:2])):
                    # Fix the online grid to a global metric lattice. Centring
                    # it on the exact spawn coordinate would shift the raster
                    # phase between episodes and create avoidable half-cell
                    # disagreement at otherwise identical world surfaces.
                    raw_origin = position[:2] - half_extent_m
                    aligned_origin = np.floor(raw_origin / resolution_m) * resolution_m
                    origin = (
                        float(aligned_origin[0]),
                        float(aligned_origin[1]),
                    )
            self._online_occupancy_grid = OnlineOccupancyGrid(
                origin_xy=origin,
                size_m=48.0,
                resolution_m=resolution_m,
            )
        return self._online_occupancy_grid

    def _episode_available_tools(
        self,
        action_space: list[str] | None,
    ) -> AvailableTools:
        """Freeze one model-visible tool surface until :meth:`reset`.

        Mutable public state is deliberately absent. If a caller attempts to
        change the declared action space mid-episode, fail instead of silently
        teaching the model a different interface.
        """

        declared = (
            None
            if action_space is None
            else frozenset(str(name) for name in action_space)
        )
        if self._frozen_available_tools is None:
            self._frozen_available_tools = resolve_available_tools(
                registered_specs=self.tools.specs,
                episode_actions=declared,
            )
            self._frozen_action_space = declared
        elif declared != self._frozen_action_space:
            raise RuntimeError(
                "Agent action space changed after the episode tool surface "
                f"was frozen: initial={sorted(self._frozen_action_space or ())}, "
                f"current={sorted(declared or ())}."
            )
        return self._frozen_available_tools

    def run(
        self,
        *,
        observation: AgentObservation,
        instruction: str,
        memory: Any = None,
        action_space: list[str] | None = None,
        perception: Any = None,
        feedback: dict[str, Any] | None = None,
        decision_requirements: DecisionRequirements | None = None,
        policy_step: int = 0,
        action_step: int = 0,
    ) -> TurnResult:
        """Run exactly one policy inference and return exactly one outcome.

        Provider transport recovery remains inside :class:`VLMClient`. Every
        policy response is parsed and validated once. A malformed decision or
        rejected state update becomes one non-terminal feedback step for the
        next policy inference; the kernel never calls the model again to repair
        semantic output inside the same turn.
        """
        requirements = decision_requirements or DecisionRequirements()
        available_tools = self._episode_available_tools(action_space)
        messages = self._build_messages(
            observation=observation,
            instruction=instruction,
            memory=memory,
            feedback=feedback,
            requirements=requirements,
            perception=perception,
            available_tools=available_tools,
        )
        context = self._build_context(
            observation=observation,
            instruction=instruction,
            memory=memory,
            available_tool_names=available_tools.names,
            perception=perception,
            feedback=feedback,
            action_step=action_step,
        )

        last_envelope: DecisionEnvelope | None = None
        last_diagnostic = ""
        turn_tool_results: list[ToolResult] = []
        turn_model_calls: list[dict[str, Any]] = []
        state_update_rejections: list[dict[str, Any]] = []
        memory_records = list(getattr(memory, "key_memories", None) or [])
        memory_update_batches: list[list[dict[str, Any]]] = []
        event_output_batches: list[dict[str, Any]] = []
        turn_planner = deepcopy(self.strategy_planner)

        def retain_event_outputs(
            envelope: DecisionEnvelope,
            validated_planner: StrategyPlanner,
        ) -> None:
            nonlocal turn_planner
            outputs = {
                field: deepcopy(getattr(envelope, field))
                for field in EVENT_WINDOW_OUTPUT_FIELDS
                if getattr(envelope, field) is not None
            }
            if outputs:
                event_output_batches.append(outputs)
                if turn_model_calls:
                    turn_model_calls[-1]["retained_event_outputs"] = deepcopy(outputs)
            # The plan cursor is durable Agent state even when this decision
            # does not revise the plan or emit another event-window output.
            # Keep it in the same transaction as the validated plan content.
            turn_planner = validated_planner

        def retain_memory_updates(
            updates: list[dict[str, Any]],
            records: list[KeyMemoryRecord],
        ) -> None:
            """Retain successful submissions until the memory owner commits."""

            nonlocal memory_records
            if not updates:
                return
            memory_records = records
            memory_update_batches.append(deepcopy(updates))
            if turn_model_calls:
                turn_model_calls[-1]["retained_memory_updates"] = deepcopy(
                    updates,
                )

        def record_call_outcome(
            outcome: str,
            *,
            correction_reason: str = "",
        ) -> None:
            """Annotate the latest private usage record with its local gate."""

            if not turn_model_calls:
                return
            turn_model_calls[-1]["decision_outcome"] = outcome
            if correction_reason:
                turn_model_calls[-1]["correction_reason"] = correction_reason

        def record_state_update_rejection(
            field: str,
            submitted_value: Any,
            reason: str,
        ) -> None:
            """Record one rejected mutation for next-decision feedback."""

            rejection = {
                "field": field,
                "submitted_value": deepcopy(submitted_value),
                "reason": reason,
            }
            state_update_rejections.append(rejection)
            if turn_model_calls:
                trace_rejections = turn_model_calls[-1].setdefault(
                    "state_update_rejections", []
                )
                trace_rejections.append(deepcopy(rejection))

        # 1. VLM call. Transport retries happen inside VLMClient; this method
        # never asks the policy to regenerate a semantic decision in-place.
        try:
            set_request_context = getattr(self.vlm, "set_request_context", None)
            if callable(set_request_context):
                set_request_context(policy_step=policy_step, action_step=action_step)
            response = self.vlm.chat(messages)
        except Exception as exc:  # noqa: BLE001
            failed_call = getattr(exc, "model_call", None)
            if isinstance(failed_call, dict):
                turn_model_calls.append(dict(failed_call))
            return TurnResult(
                status="error",
                actions=[],
                diagnostic=f"vlm_call_failed: {exc}",
                failure_kind=str(
                    getattr(
                        exc,
                        "vlm_failure_kind",
                        "model_request_failed",
                    )
                ),
                envelope=last_envelope,
                tool_results=turn_tool_results,
                model_calls=turn_model_calls,
            )

        model_call = response.get("model_call")
        if isinstance(model_call, dict):
            turn_model_calls.append(dict(model_call))

        raw = response.get("content") or ""
        # 2. Parse the single response. A rejection is surfaced as the
        # next decision's feedback without executing a tool.
        try:
            envelope = parse_decision(
                raw,
                action_horizon=self.protocol.action_horizon_limit,
                tool_argument_names={
                    spec.name: frozenset(
                        (spec.input_schema.get("properties") or {}).keys()
                    )
                    for spec in self.tools.specs
                },
            )
        except DecisionParseError as exc:
            finish_reason = response.get("finish_reason")
            last_diagnostic = (
                f"decision_parse_failed: {exc}; finish_reason={finish_reason!r}"
            )
            record_call_outcome(
                "policy_rejected",
                correction_reason=last_diagnostic,
            )
            return TurnResult(
                status="rejected",
                actions=[],
                diagnostic=last_diagnostic,
                envelope=last_envelope,
                tool_results=turn_tool_results,
                model_calls=turn_model_calls,
            )

        # Resolve revise/forget memory targets by content before the ledger
        # validates the mutation set; remember entries need no id (the ledger
        # assigns one on commit). An unmatched mutation deliberately remains
        # unresolved so ledger validation rejects the whole decision. Silently
        # dropping a forget or converting a revise into remember changes the
        # policy's requested state transition and hides an attribution signal.
        if envelope.memory:
            from tarsphere.harness.memory.key import KeyMemoryLedger as _Ledger

            snapshot = _Ledger()
            snapshot._records = {record.memory_id: record for record in memory_records}
            resolved_memory: list[dict[str, Any]] = []
            for update in envelope.memory:
                if update.get("op") in ("revise", "forget"):
                    match = snapshot.resolve_target(update.get("note", ""))
                    if match is not None:
                        update["memory_id"] = match.memory_id
                resolved_memory.append(update)
            envelope.memory[:] = resolved_memory

        last_envelope = envelope
        missing_fields = self._missing_required_fields(
            envelope,
            requirements,
        )
        missing_fields = [
            field
            for field in missing_fields
            if not any(field in batch for batch in event_output_batches)
        ]
        if missing_fields:
            last_diagnostic = "event_required_fields_missing: " + ",".join(
                missing_fields
            )
            record_call_outcome(
                "policy_rejected",
                correction_reason=last_diagnostic,
            )
            return TurnResult(
                status="rejected",
                actions=[],
                diagnostic=last_diagnostic,
                envelope=envelope,
                tool_results=turn_tool_results,
                model_calls=turn_model_calls,
            )
        plan_updates_accepted = True
        # Every call carries the Agent-owned plan cursor. Validate it
        # before dispatch; only a completed transaction may commit it.
        try:
            candidate_planner = deepcopy(turn_planner)
            self._apply_plan_updates(
                envelope,
                policy_step=policy_step,
                planner=candidate_planner,
            )
        except (TypeError, ValueError) as exc:
            plan_updates_accepted = False
            record_state_update_rejection(
                "plan",
                self._submitted_plan_updates(envelope),
                f"{type(exc).__name__}: {exc}",
            )

        # Key memories are Agent-owned durable state. Validate the complete
        # mutation set before dispatch so a failed update cannot be skipped
        # while the camera and physical world advance.
        accepted_memory_updates: list[dict[str, Any]] = []
        candidate_memory_records = memory_records
        try:
            candidate_memory_records, effective_memory = (
                KeyMemoryLedger.validate_updates(
                    memory_records,
                    envelope.memory,
                    action_step=int(action_step),
                )
            )
        except (TypeError, ValueError) as exc:
            record_state_update_rejection(
                "memory",
                envelope.memory,
                f"{type(exc).__name__}: {exc}",
            )
        else:
            # Deduplicated restatements are valid but change nothing;
            # only effective updates are credited to the trace.
            accepted_memory_updates = effective_memory
            if len(effective_memory) < len(envelope.memory):
                if turn_model_calls:
                    turn_model_calls[-1].setdefault(
                        "output_repairs",
                        [],
                    ).append(
                        f"{len(envelope.memory) - len(effective_memory)} "
                        "memory restatement(s) deduplicated against "
                        "retained facts"
                    )

        if state_update_rejections:
            rejected_fields = sorted(
                {str(item["field"]) for item in state_update_rejections}
            )
            last_diagnostic = (
                "state_update_rejected: "
                + ",".join(rejected_fields)
                + "\n"
                + self._format_state_update_rejections(state_update_rejections)
            )
            record_call_outcome(
                "policy_rejected",
                correction_reason=last_diagnostic,
            )
            return TurnResult(
                status="rejected",
                actions=[],
                diagnostic=last_diagnostic,
                envelope=envelope,
                tool_results=turn_tool_results,
                model_calls=turn_model_calls,
            )

        block_error = self._validate_action_block(
            envelope,
            available_tool_names=available_tools.names,
        )
        if block_error:
            last_diagnostic = f"action_block_invalid: {block_error}"
            record_call_outcome(
                "policy_rejected",
                correction_reason=last_diagnostic,
            )
            return TurnResult(
                status="rejected",
                actions=[],
                diagnostic=last_diagnostic,
                envelope=envelope,
                tool_results=turn_tool_results,
                model_calls=turn_model_calls,
            )

        # Execute the ordinary prefix and the first feedback-dependent
        # call. Its fresh result ends this decision; later authored calls
        # cannot execute without another policy inference.
        boundary_index = self._feedback_boundary_index(envelope)
        executable_count = (
            len(envelope.actions) if boundary_index is None else boundary_index + 1
        )
        pending_calls = list(envelope.actions[1:executable_count])
        discarded_calls = list(envelope.actions[executable_count:])

        # 3. Resolve only the first call. RobotAgent resolves each suffix
        # call at its own action step from the fresh observation produced
        # by the prior call. A later malformed or failed call is therefore
        # that call's result and cannot retroactively cancel its prefix.
        actions, content, dispatch_results = self._dispatch_sequence(
            envelope,
            context,
            first_only=True,
        )
        turn_tool_results.extend(dispatch_results)
        result_obj = dispatch_results[-1] if dispatch_results else None
        if not actions or result_obj is None:
            raise RuntimeError(f"tool dispatch produced no runtime payload: {content}")
        is_error = bool(result_obj.is_error)
        is_planning_result = bool(
            not is_error and result_obj.metadata.get("planning_status") == "no_route"
        )
        if is_error:
            record_call_outcome("tool_failed")
            # A handler-side failure is still this decision's completed
            # tool result. The model wrote memory and event-window fields
            # before attempting the call, so retain the already validated
            # state exactly as for a successful tool result.
            retain_memory_updates(
                accepted_memory_updates,
                candidate_memory_records,
            )
            retain_event_outputs(envelope, candidate_planner)
            if plan_updates_accepted:
                self.strategy_planner.commit_from(turn_planner)
            return TurnResult(
                status="completed",
                actions=actions,
                pending_calls=pending_calls,
                discarded_calls=discarded_calls,
                envelope=envelope,
                tool_results=turn_tool_results,
                model_calls=turn_model_calls,
                accepted_memory_update_batches=memory_update_batches,
                accepted_event_output_batches=event_output_batches,
                plan_updates_accepted=plan_updates_accepted,
                diagnostic=(
                    "tool execution failed; the failure is retained as "
                    "this action-step result and the authored suffix "
                    "remains queued"
                ),
            )
        retain_memory_updates(
            accepted_memory_updates,
            candidate_memory_records,
        )
        retain_event_outputs(envelope, candidate_planner)
        record_call_outcome(
            "planning_result" if is_planning_result else "executable_decision"
        )
        if plan_updates_accepted:
            self.strategy_planner.commit_from(turn_planner)
        return TurnResult(
            status="completed",
            actions=actions,
            pending_calls=pending_calls,
            discarded_calls=discarded_calls,
            envelope=envelope,
            tool_results=turn_tool_results,
            model_calls=turn_model_calls,
            accepted_memory_update_batches=memory_update_batches,
            accepted_event_output_batches=event_output_batches,
            plan_updates_accepted=plan_updates_accepted,
        )

    def resolve_queued_call(
        self,
        *,
        call: ToolCall,
        observation: AgentObservation,
        instruction: str,
        memory: Any,
        action_space: list[str] | None,
        perception: Any,
        feedback: dict[str, Any] | None,
        action_step: int,
    ) -> TurnResult:
        """Resolve one previously authored call from this step's live state.

        This performs no model inference and accepts no new plan or memory
        output. The call was authored in an earlier policy decision, but its
        handler runs only now so camera captures, pose, memory, and timestamps
        all describe the state after every preceding call in that block.
        """

        available_tools = self._episode_available_tools(action_space)
        context = self._build_context(
            observation=observation,
            instruction=instruction,
            memory=memory,
            available_tool_names=available_tools.names,
            perception=perception,
            feedback=feedback,
            action_step=action_step,
        )
        envelope = DecisionEnvelope(
            thinking="queued call from an earlier policy decision",
            actions=[call],
        )
        actions, content, results = self._dispatch_sequence(
            envelope,
            context,
            first_only=True,
        )
        if not actions or not results:
            raise RuntimeError(
                f"queued tool dispatch produced no runtime payload: {content}"
            )
        failed = bool(results[-1].is_error)
        return TurnResult(
            status="completed",
            actions=actions,
            tool_results=results,
            diagnostic=(
                "queued tool execution failed; the failure is retained as "
                "this action-step result"
                if failed
                else ""
            ),
        )

    @staticmethod
    def _submitted_plan_updates(envelope: DecisionEnvelope) -> dict[str, Any]:
        """Return exactly the Agent-authored plan fields under validation."""

        submitted: dict[str, Any] = {}
        if envelope.plan is not None:
            submitted["plan"] = deepcopy(envelope.plan)
        if envelope.subgoal_checked:
            submitted["subgoal"] = "checked"
        return submitted

    @staticmethod
    def _format_state_update_rejections(
        rejections: list[dict[str, Any]],
    ) -> str:
        """Render exact rejection feedback for the Agent's next decision."""

        lines = [
            "The following submitted state fields were not stored. No call "
            "from that decision executed. Correct the rejected fields and "
            "resubmit the complete decision.",
        ]
        for rejection in rejections:
            lines.append(
                f"- `{rejection['field']}` rejected: "
                f"{rejection['reason']}\n"
                "- submitted value: "
                + json.dumps(
                    rejection["submitted_value"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return "\n".join(lines)

    def _build_messages(
        self,
        *,
        observation: AgentObservation,
        instruction: str,
        memory: Any,
        feedback: dict[str, Any] | None,
        requirements: DecisionRequirements,
        perception: Any = None,
        available_tools: AvailableTools,
    ) -> list[dict[str, Any]]:
        plan_summary = self.strategy_planner.active_plan_summary()
        visible_specs = list(available_tools.specs)
        locomotion_visible = bool(LOCOMOTION_TOOLS & available_tools.names)
        schema_section = envelope_schema_section(
            tools=visible_specs,
            requirements=requirements,
            action_horizon=self.protocol.action_horizon,
            visual_history_horizon=self.protocol.visual_history_horizon,
            lidar_access=(True and locomotion_visible),
            interaction_intents=self._available_public_interaction_intents(),
        )
        system = (
            self.system_preamble
            + "\n"
            + self._visual_protocol_section(
                lidar_visible=(True and locomotion_visible),
            )
            + "\n"
            + schema_section
        )

        user_content: list[dict[str, Any]] = []
        # Three model-facing layers: episode-long durable state, bounded recent
        # history, and the current observation. Official episodes acquire their
        # task through public dialogue and retain the Agent's plan; Human-free
        # component diagnostics instead disclose their explicit task here.
        if self._component_task:
            user_content.append(
                {
                    "type": "text",
                    "text": "## Component task\n" + self._component_task,
                }
            )

        # The model reads stable context first and deciding evidence last.
        longterm_parts: list[str] = []
        if plan_summary:
            longterm_parts.append("### Active plan\n" + plan_summary)
        longterm_text = self._format_longterm_state(memory)
        if longterm_text:
            longterm_parts.append("### Durable state\n" + longterm_text)
        dialogue_text = EnvelopeKernel._format_human_dialogue_history(
            memory,
            task_instruction=instruction,
            before_action_step=(
                int(getattr(memory, "action_step"))
                if getattr(memory, "action_step", None) is not None
                else None
            ),
        )
        if dialogue_text:
            longterm_parts.append(
                "### Human dialogue — full episode retention\n" + dialogue_text
            )
        if longterm_parts:
            user_content.append(
                {
                    "type": "text",
                    "text": "## Long-term memory\n" + "\n\n".join(longterm_parts),
                }
            )

        # Short-term memory: the bounded recent window.
        history_text = self._format_text_history(
            memory,
            task_instruction=instruction,
            exclude_feedback_action_step=(
                int(feedback.get("action_step", feedback.get("step")))
                if (
                    isinstance(feedback, dict)
                    and feedback.get("action_step", feedback.get("step")) is not None
                )
                else None
            ),
        )
        if history_text:
            user_content.append(
                {
                    "type": "text",
                    "text": "## Recent history\n" + history_text,
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": "## Current observation",
            }
        )
        self._append_visual_history(user_content, memory)

        # Current observation: everything measured or said at this step.
        current_parts: list[str] = []
        proprioception_lines = self._format_proprioception(
            observation,
            memory,
        )
        if proprioception_lines:
            current_parts.append(
                "### Proprioception\n" + "\n".join(proprioception_lines)
            )
        feedback_text = self._format_latest_feedback(feedback)
        if feedback_text:
            current_parts.append(
                "### Latest execution feedback — authoritative\n" + feedback_text
            )
        lidar_text = self._format_lidar(
            observation,
            locomotion_visible=locomotion_visible,
        )
        if lidar_text:
            current_parts.append(
                "### LiDAR — current measured distances\n" + lidar_text
            )
        dialogue_text = self._format_current_dialogue(
            memory,
            task_instruction=instruction,
        )
        if dialogue_text:
            current_parts.append("### Dialogue this step\n" + dialogue_text)
        # Perception results — the model's own ego sensing. This is the key
        # bridge that lets the model "see" deterministic detector facts from
        # this step and ground the semantic target it is walking toward.
        # Metric controller coordinates remain private to the relevant tool.
        perc_text = self._format_perception(perception)
        if perc_text:
            current_parts.append("### Ego perception details\n" + perc_text)
        if current_parts:
            user_content.append(
                {
                    "type": "text",
                    "text": "\n\n".join(current_parts),
                }
            )
        if not user_content:
            user_content.append(
                {
                    "type": "text",
                    "text": "Decide your next action.",
                }
            )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _visual_protocol_section(self, *, lidar_visible: bool) -> str:
        """Describe the RGB camera views visible to the Agent."""

        views = ", ".join(view.value for view in self.protocol.camera_views)
        modality = (
            "Frames shown to you are RGB only; no depth image is shown. "
            "Synchronized metric depth may be used by perception and "
            "manipulation tools to return body-relative measurements; "
            "world coordinates are never prompt fields."
        )
        if lidar_visible:
            modality += " Current-frame LiDAR is a separate ranging channel."
        return f"## Visual protocol\nViews in order: {views}. {modality}"

    @staticmethod
    def _format_perception(perception: Any) -> str:
        """Render the perception module's output as text for the prompt.

        Shows semantic detections (label + bbox + score). Metric depth remains
        tool-private; the Agent receives current RGB and configured RGB history.
        """
        if perception is None:
            return ""
        lines: list[str] = []
        detections = getattr(perception, "detections", None) or []
        for det in detections:
            label = getattr(det, "label", "?")
            score = getattr(det, "score", 0.0)
            box = getattr(det, "box_xyxy", None)
            if box is not None:
                lines.append(
                    f"- detected {label} (score {score:.2f}) at bbox {list(box)}"
                )
            else:
                lines.append(f"- detected {label} (score {score:.2f})")
        return "\n".join(lines)

    @staticmethod
    def _format_text_history(
        memory: Any,
        *,
        task_instruction: str = "",
        exclude_feedback_action_step: int | None = None,
    ) -> str:
        """Render a bounded digest of completed action steps.

        The immediately preceding execution outcome belongs to the Current
        observation and is omitted from the action digest. Older outcomes stay
        as compact lines. Durable task state belongs to the long-term ledgers.
        """

        if memory is None:
            return ""
        records = list(getattr(memory, "text_history", None) or [])

        observation_note_blocks: list[str] = []
        for record in records:
            note = str(getattr(record, "observation_note", "") or "").strip()
            if not note or note == "nothing to report":
                continue
            observation_note_blocks.append(
                f"[action step {record.action_step}]\n- {note}"
            )

        # One compact line per executed tool call: name + measured outcome.
        # The full result stays in the single Latest-feedback block; this
        # action history carries the measured magnitudes (actual body
        # displacement, contact) so failure and drift patterns stay visible
        # across the window at a few tokens per step.
        action_lines: list[str] = []
        for record in records:
            if (
                exclude_feedback_action_step is not None
                and int(record.action_step) == exclude_feedback_action_step
            ):
                continue
            feedback = getattr(record, "execution_feedback", None)
            if not isinstance(feedback, dict):
                continue
            name = str(
                feedback.get("call_name")
                or getattr(record, "executed_action", {}).get("action")
                or "?"
            )
            parts: list[str] = []
            motion = feedback.get("motion_outcome")
            if isinstance(motion, dict) and motion.get("available"):
                delta = motion.get("achieved_body_delta") or {}
                parts.append(
                    "moved {f:+.2f}f {l:+.2f}l {t:+.0f}\u00b0".format(
                        f=float(delta.get("forward_m") or 0.0),
                        l=float(delta.get("left_m") or 0.0),
                        t=float(delta.get("turn_deg") or 0.0),
                    )
                )
            if feedback.get("physical_contact"):
                parts.append("contact")
            if feedback.get("executed") is False:
                detail = str(feedback.get("feedback") or "").strip()
                parts.append("failed" + (f" — {detail[:80]}" if detail else ""))
            elif not parts:
                detail = str(feedback.get("feedback") or "").strip()
                if detail:
                    parts.append(f"result — {detail[:120]}")
            outcome = "; ".join(parts) if parts else "ok"
            action_lines.append(f"[step {record.action_step}] {name}: {outcome}")

        sections: list[str] = []
        if action_lines:
            sections.append("### Action history\n" + "\n".join(action_lines))
        if observation_note_blocks:
            sections.append("### Visual State\n" + "\n".join(observation_note_blocks))
        if not sections:
            return ""
        return "\n\n".join(sections)

    @staticmethod
    def _collect_human_dialogue_turns(memory: Any) -> list[Any]:
        turns: list[Any] = []
        for record in getattr(memory, "humans", None) or []:
            turns.extend(record.dialogue)
        turns.sort(key=lambda turn: (int(turn.action_step), int(turn.sequence)))
        return turns

    @staticmethod
    def _render_dialogue_who(turn: Any, *, actor_id: str) -> str:
        if turn.speaker != actor_id:
            return turn.speaker
        status = "delivered" if turn.delivered else "not delivered"
        return f"you → {turn.recipient} [{status}]"

    @staticmethod
    def _dialogue_without_task_echo(
        memory: Any,
        *,
        task_instruction: str,
    ) -> list[Any]:
        """Return public dialogue without replaying the task's source turn."""

        turns = EnvelopeKernel._collect_human_dialogue_turns(memory)
        task_key = " ".join(str(task_instruction or "").split()).casefold()
        if not task_key:
            return turns
        actor_id = getattr(memory, "actor_id", "robot")
        for index, turn in enumerate(turns):
            if turn.speaker == actor_id:
                continue
            turn_key = " ".join(str(turn.text or "").split()).casefold()
            if turn_key == task_key:
                return turns[:index] + turns[index + 1 :]
            break
        return turns

    @staticmethod
    def _format_human_dialogue_history(
        memory: Any,
        *,
        task_instruction: str = "",
        before_action_step: int | None = None,
    ) -> str:
        """Render completed-step dialogue, never the open step twice.

        ``_build_messages`` gives the open action step its own current-dialogue
        block.  History therefore owns only earlier turns.  Direct callers may
        omit ``before_action_step`` when they intentionally need the complete
        episode transcript.
        """

        actor_id = getattr(memory, "actor_id", "robot")
        turns = EnvelopeKernel._dialogue_without_task_echo(
            memory,
            task_instruction=task_instruction,
        )
        if before_action_step is not None:
            turns = [
                turn
                for turn in turns
                if int(turn.action_step) < int(before_action_step)
            ]
        if not turns:
            return ""
        blocks: list[str] = []
        current_step: int | None = None
        lines: list[str] = []
        for turn in turns:
            step = int(turn.action_step)
            if current_step != step:
                if lines:
                    blocks.append("\n".join(lines))
                current_step = step
                lines = [f"[action step {step}]"]
            who = EnvelopeKernel._render_dialogue_who(
                turn,
                actor_id=actor_id,
            )
            lines.append(f'- dialogue: {who}: "{turn.text}"')
        if lines:
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    @staticmethod
    def _format_body_motion_tracking(motion: dict[str, Any]) -> str:
        """Render threshold-free tracking evidence for one body-vector action."""

        fields: list[str] = []
        ratio = motion.get("translation_completion_ratio")
        if ratio is not None:
            fields.append(f"translation={float(ratio):+.2f}x")
        turn_error = motion.get("turn_error_deg")
        if turn_error is not None:
            fields.append(f"turn_error={float(turn_error):+.1f}°")
        if not fields:
            return ""
        return "; ".join(fields)

    @staticmethod
    def _format_physical_contact(feedback: dict[str, Any]) -> str:
        """Group public contacts without replaying raw contact dictionaries."""

        if not bool(feedback.get("physical_contact", False)):
            return ""
        contacts = feedback.get("physical_contact_details") or []
        if not contacts:
            return "contact: measured; details unavailable"

        grouped: dict[str, dict[str, Any]] = {}
        for contact in contacts:
            if not isinstance(contact, dict):
                raise TypeError("physical_contact_details entries must be mappings.")
            source = str(contact.get("source") or "contact")
            human_id = str(contact.get("human_id") or "").strip()
            if source == "human":
                label = f"human {human_id}".strip()
            elif source in {"static_scene", "scene_geometry"}:
                label = "scene geometry"
            else:
                label = source.replace("_", " ")
            entry = grouped.setdefault(
                label,
                {
                    "directions": [],
                    "distances": [],
                },
            )
            direction = str(contact.get("direction") or "").strip()
            if direction and direction not in entry["directions"]:
                entry["directions"].append(direction)
            distance = contact.get("distance_m")
            if distance is not None:
                entry["distances"].append(float(distance))

        rendered: list[str] = []
        for label, entry in grouped.items():
            item = label
            if entry["directions"]:
                item += " at " + "/".join(entry["directions"])
            if entry["distances"]:
                item += f", nearest={min(entry['distances']):.2f}m"
            rendered.append(item)
        reference = (
            "contact during navigation (recorded positions, final-body bearings): "
            if feedback.get("contact_reference")
            == "navigation_contact_positions_in_final_body_frame"
            else "contact: "
        )
        return reference + "; ".join(rendered)

    @staticmethod
    def _format_body_stability(feedback: dict[str, Any]) -> str:
        """Render current gravity tilt without assigning a recovery action."""

        body_stability = feedback.get("body_stability")
        if not body_stability:
            return ""
        if not isinstance(body_stability, dict):
            raise TypeError("body_stability must be a mapping when present.")
        gravity_tilt_deg = body_stability.get("gravity_tilt_deg")
        if gravity_tilt_deg is None:
            raise ValueError(
                "body_stability must contain gravity_tilt_deg when present."
            )
        return f"body tilt={float(gravity_tilt_deg):.1f}° from upright"

    @staticmethod
    def _format_grasp_skill(feedback: dict[str, Any]) -> str:
        """Render the postcondition of one completed manipulation skill."""

        grasp_skill = feedback.get("grasp_skill")
        if not grasp_skill:
            return ""
        if not isinstance(grasp_skill, dict):
            raise TypeError("grasp_skill must be a mapping when present.")
        hand = str(grasp_skill.get("hand") or "")
        if hand != "right":
            raise ValueError("grasp_skill must name the certified right holding hand.")
        state = {
            "arms_retracted": grasp_skill.get("arms_retracted"),
            "standing": grasp_skill.get("standing"),
            "transport_ready": grasp_skill.get("transport_ready"),
        }
        if any(not isinstance(value, bool) for value in state.values()):
            raise ValueError(
                "grasp_skill must report Boolean arms_retracted, standing, "
                "and transport_ready states."
            )
        return (
            "grasp skill: right hand is holding the target; "
            f"arms retracted={state['arms_retracted']}; "
            f"standing={state['standing']}; "
            f"transport ready={state['transport_ready']}"
        )

    @staticmethod
    def _format_grasp_failure(feedback: dict[str, Any]) -> str:
        """Render the stable public reason for an unsuccessful grasp action."""

        failure = feedback.get("grasp_failure")
        if not failure:
            return ""
        if not isinstance(failure, dict):
            raise TypeError("grasp_failure must be a mapping when present.")
        code = str(failure.get("code") or "").strip()
        message = str(failure.get("message") or "").strip()
        if code not in GRASP_FAILURE_CODES:
            raise ValueError(
                "grasp_failure.code is outside the closed runtime vocabulary: "
                f"{code!r}."
            )
        if not message:
            raise ValueError("grasp_failure.message must be non-empty.")
        return f"grasp failure `{code}`: {message}"

    @staticmethod
    def _format_grasp_reachability(feedback: dict[str, Any]) -> str:
        """Render sensor geometry and public planning outcomes after refusal."""

        reachability = feedback.get("grasp_reachability")
        if not reachability:
            return ""
        if not isinstance(reachability, dict):
            raise TypeError("grasp_reachability must be a mapping when present.")
        if reachability.get("reachable") is not False:
            raise ValueError("grasp_reachability is emitted only for a refused grasp.")
        target = reachability.get("target_observation")
        right_hand = reachability.get("right_hand")
        if not isinstance(target, dict):
            raise TypeError("grasp_reachability.target_observation must be a mapping.")
        stage = str(reachability.get("stage") or "")
        if stage != "motion_planning":
            raise ValueError(
                "grasp_reachability.stage must name the measured failure "
                "stage; arm planning is the only one that can refuse a grasp."
            )
        lines = [
            f"grasp reachability: unreachable at {stage}; "
            "target horizontal distance from ego camera="
            f"{float(target['horizontal_distance_m']):.2f} m, "
            f"height from ego camera={float(target['height_from_camera_m']):+.2f} m, "
            f"bearing={float(target['bearing_deg']):+.1f}° "
            "(positive bearing is left)."
        ]
        if not isinstance(right_hand, dict):
            raise TypeError(
                "motion-planning reachability requires right_hand evidence."
            )
        palm_goal = right_hand.get("palm_goal_pelvis_m")
        if palm_goal is not None:
            if not isinstance(palm_goal, dict):
                raise TypeError(
                    "grasp_reachability.right_hand.palm_goal_pelvis_m must be "
                    "a mapping when present."
                )
            forward = palm_goal["forward_m"]
            lateral = palm_goal["lateral_m"]
            vertical = palm_goal["vertical_m"]
            lines.append(
                "requested palm goal relative to pelvis: "
                f"forward={float(forward[0]):+.2f}..{float(forward[1]):+.2f} m, "
                f"lateral={float(lateral[0]):+.2f}..{float(lateral[1]):+.2f} m "
                "(positive lateral is left), "
                f"vertical={float(vertical[0]):+.2f}..{float(vertical[1]):+.2f} m "
                "(positive vertical is up)."
            )
        candidate_count = int(right_hand["candidate_count"])
        details = [
            f"right hand: {candidate_count} "
            f"{'candidate' if candidate_count == 1 else 'candidates'}",
            f"failures={right_hand['failures']}",
        ]
        lines.append("; ".join(details) + ".")
        reach_correction = right_hand.get("reach_correction")
        if reach_correction is not None:
            if not isinstance(reach_correction, dict):
                raise TypeError(
                    "grasp_reachability.right_hand.reach_correction must be "
                    "a mapping when present."
                )
            status = str(reach_correction.get("status") or "")
            basis = str(reach_correction.get("basis") or "").strip()
            if status == "measured":
                body_forward_m = float(reach_correction["body_forward_m"])
                sample_count = int(reach_correction["sample_count"])
                approach = reach_correction.get("approach")
                if not isinstance(approach, dict):
                    raise TypeError(
                        "Measured reach_correction requires an approach mapping."
                    )
                kind = str(approach.get("kind") or "unknown")
                candidate_noun = "candidate" if sample_count == 1 else "candidates"
                lines.append(
                    f"Measured reach correction: of {sample_count} probed "
                    f"failed {candidate_noun}, the median needed about "
                    f"{body_forward_m:.2f} m of movement along the "
                    f"body-forward direction to become solvable "
                    f"({kind} approach). Basis: {basis}"
                )
            elif status == "not_found_within_budget":
                budget_m = float(reach_correction["probe_budget_m"])
                lines.append(
                    "No body-forward reach correction was found within the "
                    f"planner probe budget of {budget_m:.2f} m. "
                    f"Basis: {basis}"
                )
            elif status == "unavailable":
                reason = str(reach_correction.get("reason") or "").strip()
                lines.append(
                    "Body-forward reach correction could not be measured "
                    f"({reason}). Basis: {basis}"
                )
            else:
                raise ValueError(
                    "grasp_reachability.right_hand.reach_correction has an "
                    f"unknown status: {status!r}."
                )
        measured_forward_m = None
        if isinstance(reach_correction, dict):
            if reach_correction.get("status") == "measured":
                measured_forward_m = float(reach_correction["body_forward_m"])
        if measured_forward_m is not None:
            retry_guidance = (
                "The target itself was found and measured; only the arm path "
                "from where you are now failed. Step closer by about "
                f"{measured_forward_m:.2f} m along body-forward, adjust body "
                "height if needed, and call grasp_object again. Solvability "
                "depends on stance, height, bearing, and approach — there is "
                "no single fixed distance threshold."
            )
        else:
            retry_guidance = (
                "The target itself was found and measured; only the arm path "
                "from where you are now failed. Move to a different stance, "
                "or change your body height, and call grasp_object again. "
                "Solvability depends on stance, height, bearing, and approach "
                "— there is no single fixed distance threshold."
            )
        lines.append(retry_guidance)
        return " ".join(lines)

    @staticmethod
    def _format_current_dialogue(
        memory: Any,
        *,
        task_instruction: str = "",
    ) -> str:
        """Render public utterances delivered on the open action step."""

        if memory is None:
            return ""
        actor_id = getattr(memory, "actor_id", "robot")
        current_step = int(getattr(memory, "action_step", 0) or 0)
        lines: list[str] = []
        for turn in EnvelopeKernel._dialogue_without_task_echo(
            memory,
            task_instruction=task_instruction,
        ):
            if int(turn.action_step) != current_step:
                continue
            who = EnvelopeKernel._render_dialogue_who(
                turn,
                actor_id=actor_id,
            )
            lines.append(f'- {who}: "{turn.text}"')
        return "\n".join(lines)

    @staticmethod
    def _format_longterm_state(
        memory: Any,
    ) -> str:
        """Episode-lifetime reference state: cast, landmarks, holdings.

        These survive the whole episode: the authored Human directory is
        task background; semantic landmarks are spatial assets the Agent
        established through remember_entity; hand/object state is the
        measured holding ledger.
        """
        if memory is None:
            return ""
        lines: list[str] = []
        human_lines = EnvelopeKernel._format_human_directory(memory)
        if human_lines:
            lines.append("- episode humans:")
            lines.extend(f"  {line}" for line in human_lines)
        semantic_entities = list(getattr(memory, "semantic_entities", None) or [])
        if semantic_entities:
            lines.append("- semantic landmarks you saved (shown on the map):")
            lines.extend(f"  {record.describe()}" for record in semantic_entities)
        key_memories = list(getattr(memory, "key_memories", None) or [])
        if key_memories:
            lines.append("- key facts you chose to retain:")
            lines.extend(f"  {record.describe()}" for record in key_memories)
        grasped = list(getattr(memory, "grasped_hands", None) or [])
        held_labels = {
            " ".join(str(getattr(item, "target_label", "")).split()).casefold()
            for item in grasped
            if str(getattr(item, "target_label", "")).strip()
        }
        known_objects = [
            entry
            for entry in (getattr(memory, "known_objects", None) or [])
            if " ".join(str(getattr(entry, "label", "")).split()).casefold()
            not in held_labels
        ]
        if known_objects:
            lines.append("- objects you have located:")
            lines.extend(f"  {entry.describe()}" for entry in known_objects)
        if grasped:
            lines.append("- hands: " + "; ".join(item.describe() for item in grasped))
        posture = str(getattr(memory, "body_posture", "") or "").strip()
        if posture:
            lines.append(f"- body posture: {posture}")
        tilt_deg = float(getattr(memory, "camera_gaze_tilt_deg", 0.0) or 0.0)
        if tilt_deg:
            direction = "down" if tilt_deg < 0 else "up"
            lines.append(
                f"- camera: aimed {abs(tilt_deg):.0f}° {direction} from its "
                "default navigation view; `tilt_camera(0)` returns to that view"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_human_directory(
        memory: Any,
    ) -> list[str]:
        lines: list[str] = []
        humans = sorted(
            getattr(memory, "humans", None) or [],
            key=lambda record: record.speaker,
        )
        for record in humans:
            parts: list[str] = []
            caption = str(getattr(record, "caption", "") or "").strip()
            job = str(getattr(record, "job", "") or "").strip()
            if caption:
                parts.append(f"public identity description: {caption}")
            if job:
                parts.append(f"public job: {job}")
            for fact in getattr(record, "public_profile_facts", ()):
                parts.append(f"known fact: {fact['text']}")
            utterance_step = getattr(record, "last_utterance_action_step", None)
            if utterance_step is not None:
                parts.append(
                    f"latest utterance received at step {int(utterance_step)} "
                    "(language provides no spatial position)"
                )
            seen_step = getattr(record, "last_seen_action_step", None)
            if seen_step is not None:
                parts.append(
                    f"Human last visually located at step "
                    f"{int(seen_step)}; this is historical visual evidence "
                    "and supplies no current position"
                )
            if parts:
                lines.append(f"{record.speaker}: " + "; ".join(parts))
        return lines

    def _append_visual_history(
        self,
        content: list[dict[str, Any]],
        memory: Any,
    ) -> None:
        """Append native CURRENT images or one bounded history mosaic.

        A zero visual-history horizon is a real current-only protocol: the
        native 4:3 image is sent directly, so normalized visual coordinates
        share exactly the coordinate system consumed by ego-point tools.
        Positive horizons use the labeled square mosaic.
        """

        if memory is None:
            return
        frames = list(getattr(memory, "visual_history", None) or [])
        if not frames:
            return
        steps = [int(frame.action_step) for frame in frames]
        step_list = ", ".join(str(step) for step in steps)
        current_step = steps[-1]
        for view_name in self.protocol.camera_views:
            rgb_frames = []
            for frame in frames:
                view = frame.views.get(view_name.value)
                if view is None:
                    raise RuntimeError(
                        f"action step {frame.action_step} has no requested "
                        f"{view_name.value!r} camera frame"
                    )
                if getattr(view, "rgb", None) is None:
                    raise RuntimeError(
                        f"{view_name.value!r} camera has no RGB frame at action "
                        f"step {frame.action_step}"
                    )
                rgb_frames.append(view.rgb)

            if (
                view_name.value == "ego"
                and self._open_frontier_action_step == current_step
                and (self._open_frontier_overlay_rgb is not None)
            ):
                rgb_frames[-1] = self._open_frontier_overlay_rgb

            if self.protocol.visual_history_horizon == 0:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"### Current — {view_name.value} RGB — step "
                            f"{current_step}; normalized image coordinates refer "
                            "to this complete image"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _numpy_rgb_to_data_url(rgb_frames[-1])},
                    }
                )
                continue

            chronology = f"steps [{step_list}]; CURRENT {current_step}"
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"### Temporal visual context — {view_name.value} RGB — "
                        f"{chronology}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _numpy_rgb_to_data_url(
                            _compose_visual_history_mosaic(rgb_frames, steps)
                        )
                    },
                }
            )

        if (
            self._open_frontier_action_step == current_step
            and self._open_frontier_exploration_memory_rgb is not None
        ):
            content.append(
                {
                    "type": "text",
                    "text": (
                        "### Exploration BEV — persistent occupancy map — "
                        f"step {current_step}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _numpy_rgb_to_png_data_url(
                            self._open_frontier_exploration_memory_rgb
                        )
                    },
                }
            )

    @staticmethod
    def _format_latest_feedback(feedback: dict[str, Any] | None) -> str:
        """Keep the immediately preceding tool outcome salient.

        This block is derived directly from the latest feedback object. Closed
        outcomes are omitted from the matching short-term action line, so the
        complete result appears exactly once in the next model input.
        """

        if not feedback:
            return ""
        action = str(feedback.get("action") or "unknown_action")
        lines = [f"last action: {action}"]
        if feedback.get("executed") is False:
            lines.append("status: failed; no environment action executed")
        report = str(feedback.get("feedback") or "").strip()
        report_is_structurally_covered = False
        motion = feedback.get("motion_outcome")
        if isinstance(motion, dict) and bool(motion.get("available", False)):
            report_is_structurally_covered = True
            if motion.get("kind") == "navigation_motion":
                achieved = motion["achieved_body_delta"]
                lines.append(
                    "navigation net motion: "
                    f"F={float(achieved['forward_m']):+.2f}m, "
                    f"L={float(achieved['left_m']):+.2f}m, "
                    f"Y={float(achieved['turn_deg']):+.1f}°"
                )
                motion = {}
            if motion.get("kind") == "body_motion":
                requested = motion.get("requested_body_delta") or {}
                achieved = motion.get("achieved_body_delta") or {}
                body_motion = (
                    "motion: req "
                    f"F={float(requested.get('forward_m', 0.0)):+.2f}m, "
                    f"L={float(requested.get('left_m', 0.0)):+.2f}m, "
                    f"Y={float(requested.get('turn_deg', 0.0)):+.1f}° -> got "
                    f"F={float(achieved.get('forward_m', 0.0)):+.2f}m, "
                    f"L={float(achieved.get('left_m', 0.0)):+.2f}m, "
                    f"Y={float(achieved.get('turn_deg', 0.0)):+.1f}°"
                )
                tracking = EnvelopeKernel._format_body_motion_tracking(motion)
                if tracking:
                    body_motion += f"; {tracking}"
                lines.append(body_motion)
                motion = None
        if isinstance(motion, dict) and bool(motion.get("available", False)):
            requested = motion.get("requested")
            achieved = motion.get("achieved_along_command")
            unit = str(motion.get("unit") or "")
            if requested is not None and achieved is not None:
                ratio = motion.get("completion_ratio")
                ratio_text = (
                    f"; completion={float(ratio):.2f}x" if ratio is not None else ""
                )
                lines.append(
                    f"motion: req {float(requested):.2f}{unit} -> "
                    f"got {float(achieved):.2f}{unit}{ratio_text}"
                )
        contact = EnvelopeKernel._format_physical_contact(feedback)
        if contact:
            report_is_structurally_covered = True
            lines.append(contact)
        navigation = EnvelopeKernel._format_navigation_execution(feedback)
        if navigation:
            report_is_structurally_covered = True
            lines.append(navigation)
        body_stability = EnvelopeKernel._format_body_stability(feedback)
        if body_stability:
            lines.append(body_stability)
        grasp_skill = EnvelopeKernel._format_grasp_skill(feedback)
        if grasp_skill:
            report_is_structurally_covered = True
            lines.append(grasp_skill)
        grasp_failure = EnvelopeKernel._format_grasp_failure(feedback)
        if grasp_failure:
            report_is_structurally_covered = True
            lines.append(grasp_failure)
        grasp_reachability = EnvelopeKernel._format_grasp_reachability(feedback)
        if grasp_reachability:
            report_is_structurally_covered = True
            lines.append(grasp_reachability)
        if action == "place_object" and bool(feedback.get("placed", False)):
            report_is_structurally_covered = True
            lines.append("placement: completed")
        if report and not report_is_structurally_covered:
            lines.append(f"result: {report}")
        interaction = str(feedback.get("interaction_feedback") or "").strip()
        if interaction:
            lines.append(f"interaction: {interaction}")
        return "\n".join(lines)

    @staticmethod
    def _format_navigation_execution(feedback: dict[str, Any]) -> str:
        """Render public delegated-navigation progress without hidden state."""

        navigation = feedback.get("navigation_execution")
        if not isinstance(navigation, dict) or not navigation:
            return ""
        goal = navigation["remaining_pointgoal_body_xy"]
        goal_text = (
            f"current-body waypoint F={float(goal[0]):+.2f}m/L={float(goal[1]):+.2f} m"
            if goal is not None
            else "current-body waypoint unmeasured"
        )
        before = navigation.get("distance_before_m")
        after = navigation.get("distance_after_estimate_m")
        progress = navigation.get("progress_m")
        distance_text = (
            f"{float(before):.2f}->{float(after):.2f} m"
            if before is not None and after is not None
            else "unmeasured"
        )
        progress_text = (
            f", delta={float(progress):+.2f} m" if progress is not None else ""
        )
        execution_text = ""
        if "stop_reason" in navigation:
            distance = navigation["travelled_distance_m"]
            path_text = (
                f"{float(distance):.2f}m" if distance is not None else "unmeasured"
            )
            execution_text = (
                f"; return_condition={navigation['stop_reason']}, "
                f"travelled={path_text}, "
                f"physics_steps={navigation['physics_steps']}/{navigation['physics_step_limit']}, "
                f"goal_tolerance={navigation['goal_tolerance_m']:.2f}m"
            )
            if navigation["stop_reason"] == "position_heading_window":
                execution_text += (
                    f"; window_steps={navigation['position_heading_window_steps']}, "
                    f"position_radius={navigation['position_window_radius_m']:.3f}m, "
                    f"heading_span={navigation['heading_window_span_rad']:.3f}rad"
                )
            elif navigation["stop_reason"] == "coverage_window":
                execution_text += (
                    "; steps_without_new_coverage_or_closer_distance="
                    f"{navigation['steps_without_new_coverage_or_closer_distance']}"
                )
            if navigation.get("backend_error"):
                execution_text += f"; backend error: {navigation['backend_error']}"
        return (
            "nav: "
            f"requested waypoint label={navigation.get('target_description') or 'local PointGoal'!r}, "
            f"{goal_text}, "
            f"this motion measured {distance_text}{progress_text}"
            f"{execution_text}"
        )

    @staticmethod
    def _format_proprioception(
        observation: Any,
        memory: Any,
    ) -> list[str]:
        """Body-state lines: EEF position and measured holding."""

        lines: list[str] = []
        raw = getattr(observation, "raw", None) or {}
        proprio = raw.get("proprioception") if isinstance(raw, dict) else None
        if isinstance(proprio, dict):
            eef = proprio.get("eef_position_body_m")
            rotvec = proprio.get("eef_rotation_body_rad")
            gripper = proprio.get("eef_gripper_closure")
            if (
                isinstance(eef, list)
                and len(eef) == 3
                and isinstance(rotvec, list)
                and len(rotvec) == 3
            ):
                # Same 7D shape and units the move_end_effector delta uses:
                # [X, Y, Z, dRx, dRy, dRz, Gripper]; rotation is the
                # body-frame rotation vector (rad), gripper 0 open-1 closed.
                values = [
                    *eef,
                    *rotvec,
                    gripper if isinstance(gripper, (int, float)) else 0.0,
                ]
                lines.append(
                    "right hand EEF state [X, Y, Z, dRx, dRy, dRz, "
                    "Gripper] (m; rad; 0 open-1 closed): "
                    + ", ".join(f"{float(v):+.2f}" for v in values)
                )
        grasped = list(getattr(memory, "grasped_hands", None) or [])
        lines.append("right hand: holding" if grasped else "right hand: empty")
        return lines

    def _format_lidar(
        self,
        observation: AgentObservation,
        *,
        locomotion_visible: bool,
    ) -> str:
        """Render current LiDAR only while the Agent can choose locomotion."""

        if not locomotion_visible:
            return ""
        payload = (observation.raw or {}).get("lidar")
        if payload is None:
            raise RuntimeError(
                "LiDAR access is enabled for locomotion but the current "
                "observation has no lidar payload."
            )
        if not isinstance(payload, dict):
            raise TypeError("observation.raw.lidar must be a mapping.")
        return describe_lidar(LidarScan.from_mapping(payload))

    def _build_context(
        self,
        *,
        observation: AgentObservation,
        instruction: str,
        memory: Any,
        available_tool_names: frozenset[str],
        perception: Any,
        feedback: dict[str, Any] | None,
        action_step: int,
    ) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "observation": observation,
            "instruction": instruction,
            "memory": memory,
            "perception": perception,
            "available_tool_names": available_tool_names,
            "feedback": feedback or {},
            "action_step": int(action_step),
            "visual_modality": "rgb",
        }
        if bool(
            available_tool_names.intersection(
                {"navigate_to_pointgoal", "navigate_to_frontier", "navigate_to_entity"}
            )
        ):
            ctx["online_occupancy_grid"] = self._ensure_online_occupancy_grid(
                observation,
            )
            ctx["online_map_error"] = self._online_map_error
            ctx["online_map_action_step"] = self._online_map_action_step
            import os

            record_dir = os.environ.get("TARSPHERE_RECORD_DIR", "")
            if record_dir and _navigation_debug_artifacts_enabled():
                ctx["record_dir"] = record_dir
        return ctx

    @staticmethod
    def _missing_required_fields(
        envelope: DecisionEnvelope,
        requirements: DecisionRequirements,
    ) -> list[str]:
        """Find structurally absent public-event outputs.

        This validates only the harness/model interface. Semantic correctness
        remains evaluator-owned and must never be repaired from private GT.
        """

        if requirements.planning is not None and envelope.plan is None:
            return ["plan"]
        return []

    def _apply_plan_updates(
        self,
        envelope: DecisionEnvelope,
        *,
        policy_step: int,
        planner: StrategyPlanner | None = None,
    ) -> None:
        """Apply the latest Agent-authored plan or adaptation revision."""

        target = planner or self.strategy_planner

        if envelope.plan is not None:
            # A complete plan submission activates step 0 automatically and
            # replaces any active plan (adaptation).
            target.ingest(envelope.plan, policy_step=policy_step)
        elif not envelope.subgoal_checked:
            target.cancel_terminal_confirmation()
        if envelope.subgoal_checked:
            if target.active_plan is None:
                raise ValueError(
                    "subgoal checked cannot be submitted before an Agent plan exists."
                )
            target.declare_subgoal_checked(policy_step=policy_step)

    def _validate_action_block(
        self,
        envelope: DecisionEnvelope,
        *,
        available_tool_names: frozenset[str],
    ) -> str:
        """Validate only whole-program constraints before the first call.

        Tool existence, availability, and arguments are checked only when
        that exact call reaches its own serial action step. Feedback-dependent
        tools are accepted in an authored block; execution includes the
        ordinary prefix and first such tool, then discards the remaining
        suffix so the result reaches the next policy decision.
        """

        del available_tool_names
        has_feedback_boundary = self._feedback_boundary_index(envelope) is not None
        expected_length = self.protocol.action_horizon_limit
        if (
            not has_feedback_boundary
            and not self.protocol.dynamic_action_horizon
            and len(envelope.actions) != expected_length
        ):
            return (
                "ordinary policy decisions must contain exactly "
                f"{expected_length} executable calls for Action Horizon "
                f"{expected_length}; received {len(envelope.actions)}"
            )

        return ""

    def _feedback_boundary_index(
        self,
        envelope: DecisionEnvelope,
    ) -> int | None:
        """Return the first call that must be followed by policy inference."""

        singleton_tools = {
            "grasp_object",
            "navigate_to_pointgoal",
            "navigate_to_frontier",
            "navigate_to_entity",
            "report_done",
        }
        for index, call in enumerate(envelope.actions):
            spec = self.tools.spec(call.tool)
            if (
                spec is not None and spec.execution_kind == "information"
            ) or call.tool in singleton_tools:
                return index
        return None

    def _dispatch_sequence(
        self,
        envelope: DecisionEnvelope,
        context: dict[str, Any],
        *,
        first_only: bool = False,
    ) -> tuple[list[dict[str, Any]], str, list[ToolResult]]:
        """Resolve the decision's calls and return their runtime payloads.

        ``first_only`` validates and resolves only the current call;
        RobotAgent retains the authored suffix and resolves each later call
        from the observation produced by its predecessor. The full-dispatch
        mode remains for focused transaction tests.
        """
        expected_length = self.protocol.action_horizon_limit
        if (
            not self.protocol.dynamic_action_horizon
            and len(envelope.actions) > expected_length
        ):
            raise ValueError(
                "a policy decision contains at most "
                f"{expected_length} call(s) for Action Horizon "
                f"{expected_length}; received {len(envelope.actions)}"
            )
        calls_to_execute = envelope.actions[:1] if first_only else envelope.actions
        specs = [self.tools.spec(call.tool) for call in calls_to_execute]
        unknown = [
            (index, call)
            for index, (call, spec) in enumerate(
                zip(calls_to_execute, specs, strict=True)
            )
            if spec is None
        ]
        if unknown:
            index, call = unknown[0]
            failure = tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                f"unknown tool(s): {[item.tool for _, item in unknown]}",
            )
            return (
                self._runtime_actions(call, index, failure),
                failure.content,
                [failure],
            )
        available = context.get("available_tool_names")
        if not isinstance(available, frozenset):
            raise TypeError(
                "Envelope dispatch requires the resolved available_tool_names."
            )
        unavailable = [
            (index, calls_to_execute[index], spec.name)
            for index, spec in enumerate(specs)
            if spec is not None and spec.name not in available
        ]
        if unavailable:
            index, call, _name = unavailable[0]
            names = [name for _, _, name in unavailable]
            failure = tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                f"calls {names} are unavailable in the run-configured tool surface",
            )
            return (
                self._runtime_actions(call, index, failure),
                failure.content,
                [failure],
            )

        actions: list[dict[str, Any]] = []
        results: list[ToolResult] = []
        # Validate only the call that executes at this boundary. A malformed
        # suffix is reported when its own serial action step is reached.
        for index, call in enumerate(calls_to_execute):
            invalid = self.tools.validate(call)
            if invalid is not None:
                return (
                    self._runtime_actions(call, index, invalid),
                    (f"actions[{index}] {call.tool!r} failed: {invalid.content}"),
                    [invalid],
                )

        for index, call in enumerate(calls_to_execute):
            result = self.tools.execute(call, context)
            results.append(result)
            if result.is_error:
                return (
                    self._runtime_actions(call, index, result),
                    (f"actions[{index}] {call.tool!r} failed: {result.content}"),
                    results,
                )
            actions.extend(self._runtime_actions(call, index, result))

        attempted_effects: list[tuple[ToolCall, ToolResult]] = []
        committing_call: ToolCall | None = None
        try:
            for call, result in zip(calls_to_execute, results, strict=True):
                effect = result.deferred_effect
                if effect is None:
                    continue
                committing_call = call
                # Register the effect before entering commit: a well-behaved
                # transactional effect can then undo even a partial mutation
                # if its own commit raises midway through.
                attempted_effects.append((call, result))
                effect.commit()
        except Exception as exc:  # noqa: BLE001 - converted to tool failure
            rollback_errors: list[str] = []
            for _, committed_result in reversed(attempted_effects):
                try:
                    effect = committed_result.deferred_effect
                    if effect is None:
                        raise RuntimeError(
                            "committed tool result lost its deferred effect"
                        )
                    effect.rollback()
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(
                        f"{type(rollback_exc).__name__}: {rollback_exc}"
                    )
            if rollback_errors:
                raise RuntimeError(
                    "Action-block rollback failed after a deferred tool effect "
                    f"could not commit: {rollback_errors}"
                ) from exc
            if committing_call is None:
                raise RuntimeError(
                    "Deferred effect commit failed without an owning call."
                ) from exc
            failed_call = committing_call
            failure = unexpected_tool_failure(failed_call, exc)
            attempted_by_result = {
                id(attempted_result): attempted_call
                for attempted_call, attempted_result in attempted_effects
            }
            # The block is fail-closed: every earlier "saved" bookkeeping
            # result was rolled back with it, so the trace must not keep a
            # success row that never landed. Replace each attempted result
            # with an explicit failure naming the real outcome.
            annotated_results: list[ToolResult] = []
            for result_obj in results:
                owning_call = attempted_by_result.get(id(result_obj))
                if owning_call is None:
                    annotated_results.append(result_obj)
                    continue
                annotated_results.append(
                    unexpected_tool_failure(
                        owning_call,
                        RuntimeError(
                            f"{owning_call.tool} reported success but its effect "
                            "was rolled back because a later call in the same "
                            f"block failed ({type(exc).__name__}); nothing was "
                            "saved."
                        ),
                    )
                )
            annotated_results.append(failure)
            failed_index = next(
                index
                for index, item in enumerate(envelope.actions)
                if item is failed_call
            )
            return (
                self._runtime_actions(failed_call, failed_index, failure),
                (f"deferred effect for {failed_call.tool!r} failed: {failure.content}"),
                annotated_results,
            )
        if not actions:
            raise RuntimeError(
                "tool dispatch produced no runtime payload after a successful "
                "handler sequence"
            )
        return actions, "sequence resolved", results

    @staticmethod
    def _runtime_actions(
        call: ToolCall,
        index: int,
        result: ToolResult,
    ) -> list[dict[str, Any]]:
        """Annotate one tool's runtime payload for the runner."""

        if result.action is None and not result.actions:
            raise RuntimeError(f"{call.tool} produced no runtime payload")
        payloads = (
            [dict(item) for item in result.actions]
            if result.actions
            else [dict(result.action)]
        )
        annotated: list[dict[str, Any]] = []
        for payload in payloads:
            payload["call_name"] = call.tool
            payload["tool_args"] = dict(call.args)
            payload["tool_call_index"] = index
            if result.is_error:
                payload["is_error"] = True
            annotated.append(payload)
        return annotated

    @staticmethod
    def _format_info_result(result: Any) -> str:
        """Render an information call's result as prompt feedback.

        Metric grounding remains private to manipulation. The Agent receives
        only the matched RGB box and annotated current frame.
        """
        meta = getattr(result, "metadata", {}) or {}
        name = getattr(result, "name", "call")
        if meta.get("bbox_2d") is not None:
            # The handler owns the public projection. In particular,
            # locate_object includes synchronized body-relative depth while
            # keeping its private world coordinate out of this text.
            return str(getattr(result, "content", "") or f"{name}: located")
        if meta.get("detected") is False:
            target = meta.get("target", "object")
            return (
                f'{name}: "{target}" was not detected in the current ego view. '
                "The current visual evidence is insufficient; do not repeat "
                "the same call without changed evidence."
            )
        # Generic fallback: dump the content.
        return f"{name}: {getattr(result, 'content', '')}"

    @staticmethod
    def _format_tool_failure(result: ToolResult) -> str:
        """Return neutral, failure-class-aware feedback for the same turn."""

        metadata = dict(result.metadata or {})
        reason = str(metadata.get("error_type") or "")
        if reason in ARGUMENT_FAILURES:
            implication = (
                "The call arguments or required public fact must change before "
                "this call can execute."
            )
        elif reason in WORLD_FAILURES:
            implication = (
                "The measured world state refused this call; repeating it "
                "unchanged against the same evidence will refuse again."
            )
        elif reason in BACKEND_FAILURES:
            implication = (
                "The requested backend is unavailable or failed; do not treat "
                "this as object or geometry evidence."
            )
        else:
            implication = (
                "The call failed without a recognized public failure class; "
                "reassess the current evidence."
            )
        text = (
            f"The call failed with `{reason or 'unknown'}`: {result.content} "
            f"{implication}"
        )
        reachability = metadata.get("grasp_reachability")
        if reachability is not None:
            if not isinstance(reachability, dict):
                raise TypeError("grasp_reachability tool evidence must be a mapping.")
            text += "\nStructured `grasp_reachability` evidence:\n" + json.dumps(
                reachability, ensure_ascii=False, sort_keys=True
            )
        return text

    @staticmethod
    def _attach_result_images(
        content: list[dict[str, Any]],
        result: Any,
    ) -> list[dict[str, Any]]:
        """Append a tool result's own images to its feedback blocks.

        A refused call carries the same visual evidence as a successful one —
        the segmentation the refusal was measured against. Rendering only its
        text would make the model argue with a picture it cannot see.
        """
        for image_url in getattr(result, "images", None) or ():
            if image_url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": str(image_url)},
                    }
                )
        return content

    @classmethod
    def _format_info_result_content(cls, result: Any) -> list[dict[str, Any]]:
        """Render info-tool feedback as multimodal user content blocks."""
        return cls._attach_result_images(
            [{"type": "text", "text": cls._format_info_result(result)}],
            result,
        )


def _compose_visual_history_mosaic(
    arrays: list[Any],
    action_steps: list[int],
    *,
    canvas_width: int = 640,
) -> Any:
    """Compose current plus up to four complete historical frames.

    The default result is 640x640: four complete 4:3 history slots above one
    large 4:3 current frame. The square outer canvas scales with
    ``canvas_width`` and is deliberately fitted to the 1+4 composition. Labels
    live in dedicated bands, so frames and text never overlap.
    """

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if not arrays or len(arrays) != len(action_steps):
        raise ValueError("visual mosaic requires one action step per frame")
    if len(arrays) > 5:
        raise ValueError(
            "visual mosaic supports current plus at most four history frames"
        )
    if canvas_width < 320 or canvas_width % 16:
        raise ValueError("visual mosaic width must be at least 320 and divisible by 16")

    images = []
    for arr in arrays:
        array = np.asarray(arr)
        if array.dtype != np.uint8:
            array = (
                (array * 255).clip(0, 255).astype(np.uint8)
                if float(array.max()) <= 1.0
                else array.clip(0, 255).astype(np.uint8)
            )
        images.append(Image.fromarray(array).convert("RGB"))

    canvas_height = canvas_width
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    current_height = canvas_width * 3 // 4
    history_width = canvas_width // 4
    history_height = round(history_width * 3 / 4)
    total_label_height = canvas_height - current_height - history_height
    history_label_height = total_label_height // 2
    current_label_height = total_label_height - history_label_height
    label_inset = max(4, round(canvas_width / 180))
    label_font_size = max(10, round(canvas_width * 0.015))
    try:
        label_font = ImageFont.truetype(
            "DejaVuSans.ttf",
            label_font_size,
        )
    except OSError:
        label_font = ImageFont.load_default()

    def paste_contained(
        image: Any,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> None:
        scale = min(width / image.width, height / image.height)
        size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        fitted = image.resize(size, resampling)
        x = left + (width - size[0]) // 2
        y = top + (height - size[1]) // 2
        canvas.paste(fitted, (x, y))
        draw.rectangle(
            (x, y, x + size[0] - 1, y + size[1] - 1),
            outline=(92, 101, 116),
            width=1,
        )

    histories = list(zip(images[:-1], action_steps[:-1]))
    history_left = 0
    history_label_top = 0
    history_image_top = history_label_top + history_label_height
    for index, (image, action_step) in enumerate(histories):
        slot_left = history_left + index * history_width
        draw.text(
            (slot_left + label_inset, history_label_top + 1),
            f"HISTORY {index + 1} | step {action_step}",
            fill=(226, 230, 237),
            font=label_font,
        )
        paste_contained(
            image,
            left=slot_left,
            top=history_image_top,
            width=history_width,
            height=history_height,
        )

    current_left = 0
    current_label_top = history_image_top + history_height
    current_top = current_label_top + current_label_height
    current_width = canvas_width
    draw.text(
        (current_left + label_inset, current_label_top + 2),
        f"CURRENT | step {action_steps[-1]}",
        fill=(255, 255, 255),
        font=label_font,
    )
    paste_contained(
        images[-1],
        left=current_left,
        top=current_top,
        width=current_width,
        height=current_height,
    )
    return np.asarray(canvas)


def _numpy_rgb_to_data_url(arr: Any) -> str:
    import numpy as np
    from PIL import Image

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (
            (a * 255).clip(0, 255).astype(np.uint8)
            if float(a.max()) <= 1.0
            else a.astype(np.uint8)
        )
    img = Image.fromarray(a)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _numpy_rgb_to_png_data_url(arr: Any) -> str:
    """Encode annotated diagrams losslessly so labels and arrows stay sharp."""

    import numpy as np
    from PIL import Image

    array = np.asarray(arr)
    if array.dtype != np.uint8:
        array = (
            (array * 255).clip(0, 255).astype(np.uint8)
            if float(array.max()) <= 1.0
            else array.astype(np.uint8)
        )
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


__all__ = ["EnvelopeKernel", "BASE_SYSTEM_PREAMBLE"]
