"""Native skeletal-animation executor for scene NPCs.

``omni.anim.people`` provides walking, idle, looking and sitting, with NavMesh
pathfinding and runtime command injection. Initialization requires the loaded
extension, retargeted biped character assets beside the configured IsaacPeople
directory, and a baked scene NavMesh. Missing resources raise an error."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from tarsphere.humans.runtime_capabilities import HUMAN_RUNTIME_CAPABILITIES
from tarsphere.infra.control.human_body import (
    HumanBodyResult,
    HumanBodyState,
    PoseSyncCallback,
)
from tarsphere.infra.control.physical_contracts import (
    HUMAN_GOTO_ARRIVAL_TOLERANCE_M,
)
from tarsphere.infra.control.human_seating import (
    SEAT_STANCE_TOLERANCE_M,
    SEAT_YAW_TOLERANCE_RAD,
    yaw_error_rad,
)

logger = logging.getLogger(__name__)


class AnimPeopleBodyExecutor:
    """Body executor backed by ``omni.anim.people``.

    This executor sends real commands (GoTo, Idle, LookAround) through the
    ``omni.anim.people`` command injection interface.  The extension handles
    skeletal animation, NavMesh navigation, and obstacle avoidance.

    Usage::

        executor = AnimPeopleBodyExecutor(
            pose_sync=my_sync_callback,
            character_assets_path='/path/to/Isaac/People/Characters',
        )
        executor.reset(episode.humans)
        executor.apply_action('visitor', {'command': 'goto', 'target': [3, 1, 0]})
    """

    SPEECH_COMMANDS = {"speak", "ask", "report_done"}

    # IRA custom commands backed by registered SkelAnimation clips.  Each entry
    # maps a TARSphere command name to the official omni.anim.people CustomCommandName.
    # These commands require the clip to be registered into IRA's anim graph at
    # Set Up Simulation time (see ``_register_custom_commands``); injection fails
    # loud if the running anim graph does not recognize the command.
    CUSTOM_ANIMATED_COMMANDS = HUMAN_RUNTIME_CAPABILITIES.custom_animation_commands
    DEFAULT_SIT_DURATION_SEC = 5.0
    DEFAULT_CUSTOM_DURATION_SEC = 5.0
    # Ambient motion is renewed only after IRA consumes this finite command.
    # It never occupies the authored-action queue and injection can preempt it.
    AMBIENT_LOOK_AROUND_DURATION_SEC = 30.0

    # How far ``Sit.skelanim`` carries the pelvis backwards, measured off the
    # shipped clip (pelvis Y 0.957 -> 0.610, Z -0.014 -> -0.411 in the biped's
    # own Y-up frame).  The clip achieves this through joint rotation alone --
    # the character root never translates -- so the root must be authored on the
    # FLOOR, this far in front of the seat, for the body to land on the seat.
    # Authoring the seat top here instead stands the character on the furniture.
    PELVIS_SIT_BACK_TRAVEL_M = 0.40

    # ``walk_to_rot`` is consumed as the character's final world rotation and is
    # compared through ``Utils.convert_to_angle``, so it is a character yaw
    # rather than a bare heading.  Measured on steady walk segments of
    # ``human_sit_3007b22c``: course - yaw = -90 deg, i.e. yaw = heading + 90.
    CHARACTER_FACING_YAW_OFFSET_DEG = 90.0

    # Command deduplication is deliberately stricter than physical arrival.
    # It answers “is this the same requested target?”, not “has the actor
    # arrived?”.  Arrival itself uses the shared Isaac People contract.
    GOTO_COMMAND_DEDUP_TOLERANCE_M = 0.10

    def __init__(
        self,
        *,
        pose_sync: Optional[PoseSyncCallback] = None,
        character_assets_path: str = "",
        default_character: str = "F_Medical_01",
        capture_native_motion_trace: bool = False,
    ):
        if not isinstance(capture_native_motion_trace, bool):
            raise TypeError("capture_native_motion_trace must be bool.")
        self._capture_native_motion_trace = capture_native_motion_trace
        self._last_native_motion_trace_tick: Dict[str, int] = {}
        self._pose_sync = pose_sync
        self._character_assets_path = character_assets_path
        self._default_character = default_character
        self._states: Dict[str, HumanBodyState] = {}
        self._character_prim_paths: Dict[str, str] = {}
        self._character_skelroot_paths: Dict[str, str] = {}
        self._spawn_epoch = 0
        self._scene_bound_ids: set[str] = set()
        # IDs identify one injected batch across IRA's asynchronous callbacks.
        # A preempted command's late end event must not complete its successor.
        self._pending_command_ids: Dict[str, set[str]] = {}
        # Keep preempted IDs until their native end event so a Sit stand-up
        # transition remains in the replay even after GoTo replaces the batch.
        self._issued_command_ids: set[str] = set()
        self._command_serial = 0
        self._command_end_subscription = None
        self._command_start_subscription = None
        # Physics ticks are provided by the environment at native tick
        # boundaries; IRA time_code is recorded separately, never converted.
        self._physics_step: Optional[int] = None
        self._replay_log: list[Dict[str, Any]] = []
        self._last_recorded_state: Dict[str, tuple] = {}
        self._last_recorded_graph_state: Dict[str, dict] = {}
        # Last GoTo destination injected per character (x, y).  Used by the
        # injection-level dedupe so a stateless policy re-deciding "move to X"
        # every step cannot grow the omni.anim.people command queue with
        # duplicate / degenerate GoTos.  Cleared on arrival and on reset.
        self._last_injected_goto: Dict[str, tuple[float, float]] = {}
        # A Robot-aware A* route is executed as ordered GoTo legs.  AnimPeople
        # accepts one destination per command, so retaining the remaining
        # waypoints here is the only way to preserve the planned obstacle
        # avoidance instead of silently collapsing it to the final target.
        self._goto_waypoint_queues: Dict[str, list[list[float]]] = {}
        self._people_api: Any = None
        self._stage: Any = None
        self._initialized = False
        # One-shot: whether a stage attach has already let omni.anim.graph.core
        # build its CharacterManager.  Keyed on the root layer identifier, the
        # only stage identity an attach preserves -- see _stage_identity().
        # IsaacAutonomousEnv performs the attach and calls
        # mark_character_runtime_prepared(); spawning only checks the contract.
        self._character_runtime_prepared = False
        self._prepared_stage_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_pose_sync(self, pose_sync: Optional[PoseSyncCallback]):
        self._pose_sync = pose_sync

    def set_character_assets_path(self, character_assets_path: str):
        self._character_assets_path = character_assets_path

    def despawn_all(self) -> None:
        """Delete every spawned character prim from the stage.

        Called at the top of :meth:`reset` so a persistent worker does not leak
        the previous episode's characters when the next episode uses different
        human ids (``_spawn_character`` only deletes the *same* id it is about to
        create).  No-op when nothing has been spawned yet.  This is cleanup, so
        per-character failures are warned and swallowed — one stale prim must not
        block the next episode.
        """
        human_ids = [
            human_id
            for human_id in self._states
            if human_id not in self._scene_bound_ids
        ]
        if not human_ids:
            return
        try:
            from isaacsim.replicator.agent.core.agent_manager import AgentManager
            from isaacsim.replicator.agent.core.stage_util import CharacterUtil
        except Exception as exc:
            print(
                f"[WARN] Cannot despawn characters (agent core unavailable): {exc}",
                flush=True,
            )
            return
        for human_id in human_ids:
            try:
                # USD deletion does not unregister IRA's cached BehaviorScript.
                AgentManager.get_instance().deregister_agent(human_id)
                CharacterUtil.delete_character_prim(human_id)
            except Exception as exc:
                print(
                    f'[WARN] Failed to despawn character "{human_id}": {exc}',
                    flush=True,
                )

    def _quiesce_all(self) -> None:
        """Finish native locomotion/standing before removing animated actors.

        omni.anim.people's motion-matching keeps a live BezierSpline trajectory
        while a character is mid-GoTo.  Deleting the character prim while that
        trajectory is active can leave the C++ anim runtime referencing freed
        spline data, and a later anim-graph update segfaults in
        ``BezierSpline::NearestPointOnCurve`` (observed on the second episode of
        a persistent worker after a soft reset, where the respawned character's
        runtime is more fragile than a first episode's).  Injecting Idle and
        pumping Kit updates lets each walking character release its path and
        settle to idle first, so the subsequent despawn + respawn is safe.
        A held Sit also owns a transform-writing command. Preempt it through
        the official standing phase and observe the replacement Idle start.
        These are scene-setup updates, outside the new episode physics clock.
        """
        active_ids = [
            human_id
            for human_id, state in self._states.items()
            if getattr(state, "body_status", "") != "anim_people_pending"
            and getattr(state, "active_action", "") in {"goto", "sit"}
            and getattr(state, "action_status", "") == "ongoing"
        ]
        if not active_ids:
            return
        for human_id in active_ids:
            self._inject_command(human_id, f"{human_id} Idle 5")
        remaining = set(active_ids)
        for _ in range(180):
            self._pump_app_updates(count=1)
            for human_id in tuple(remaining):
                instance = self._fetch_character_script_instance(human_id)
                current = getattr(instance, "current_command", None)
                if current is not None and current.get_command_name() == "Idle":
                    remaining.remove(human_id)
            if not remaining:
                break
        if remaining:
            raise RuntimeError(
                f"Native Human commands did not quiesce before reset: {sorted(remaining)}"
            )
        print(
            f"[INFO] Quiesced {len(active_ids)} animated character(s) before despawn.",
            flush=True,
        )

    def reset(self, humans: Iterable[Any]):
        removed_skelroots = list(self._character_skelroot_paths.values())
        self._quiesce_all()
        self.despawn_all()
        self._flush_despawned_characters(removed_skelroots)
        # Fabric/native animation can retain a transform by SkelRoot path even
        # after get_character(path) briefly reports None. Never reuse that path.
        self._spawn_epoch = getattr(self, "_spawn_epoch", 0) + 1
        self._states.clear()
        self._character_prim_paths.clear()
        self._character_skelroot_paths.clear()
        self._scene_bound_ids.clear()
        self._pending_command_ids.clear()
        self._issued_command_ids.clear()
        self._replay_log.clear()
        # Setup callbacks precede the environment binding the new episode's
        # physics tick. They must not inherit the previous episode's clock.
        self._physics_step = None
        self._last_recorded_state.clear()
        self._last_recorded_graph_state.clear()
        self._last_native_motion_trace_tick.clear()
        self._last_injected_goto.clear()
        self._goto_waypoint_queues = {}
        humans = list(humans)
        for human in humans:
            human_id = str(getattr(human, "id"))
            position = [float(item) for item in getattr(human, "start_position")]
            self._states[human_id] = HumanBodyState(
                human_id=human_id,
                position=position,
                body_status="anim_people_pending",
                physical_status="stage_pending",
            )

        spawned_humans = [
            human
            for human in humans
            if not str(getattr(human, "scene_person_prim_path", "") or "").strip()
        ]
        if spawned_humans:
            self._initialize_people_api()

        for human in humans:
            human_id = str(getattr(human, "id"))
            position = [float(item) for item in getattr(human, "start_position")]
            source_path = str(
                getattr(human, "scene_person_prim_path", "") or ""
            ).strip()
            if source_path:
                self._bind_scene_person(human, source_path)
                continue
            self._spawn_character(
                human_id,
                position,
                character_name=getattr(human, "character_name", None),
                start_yaw_rad=float(getattr(human, "start_yaw", 0.0) or 0.0),
            )
        self._initialize_character_scripts(
            [str(getattr(human, "id")) for human in spawned_humans]
        )
        # Ambient LookAround gives stationary actors a live pose without
        # fabricating an authored action or blocking a later GoTo/Sit.
        for human in humans:
            human_id = str(getattr(human, "id"))
            if human_id in self._scene_bound_ids:
                state = self._states[human_id]
                state.animation_state = "native_static_pose"
                state.action_status = "idle"
                state.active_action = ""
                continue
            state = self._states[human_id]
            self._start_ambient_look_around(human_id, state)
            state.action_status = "idle"
            state.active_action = ""
        if humans:
            self._pump_app_updates(count=5)
        for human in humans:
            self._sync(str(getattr(human, "id")))

    def _flush_despawned_characters(self, skelroot_paths: list[str]) -> None:
        """Observe native removal before reusing a character's USD path.

        Delete and recreate within one Kit update can leave AnimGraph's old
        world transform alive. In a measured warm reset after Sit, the new
        actor retained the previous chair's interact position. This setup-only
        update separates removal from creation and verifies native detachment;
        it never substitutes an authored transform for a measured pose.
        """
        if not skelroot_paths:
            return
        import omni.anim.graph.core as ag
        from omni.metropolis.utils.simulation_util import SimulationUtil

        remaining = list(skelroot_paths)
        for _ in range(4):
            self._pump_app_updates(count=1)
            remaining = [
                path
                for path in remaining
                if (
                    ag.get_character(path) is not None
                    or SimulationUtil.get_agent_script_instance_by_path(path)
                    is not None
                )
            ]
            if not remaining:
                return
        raise RuntimeError(
            "AnimGraph or BehaviorScript retained deleted Human characters before respawn: "
            f"{remaining}. Refusing to reuse stale native poses."
        )

    def get_state(self, human_id: str) -> Optional[HumanBodyState]:
        return self._states.get(human_id)

    def all_states(self) -> Dict[str, HumanBodyState]:
        return dict(self._states)

    def set_physics_step(self, physics_step: int) -> None:
        """Bind following native callbacks/state observations to this tick."""
        if (
            isinstance(physics_step, bool)
            or int(physics_step) != physics_step
            or physics_step < 0
        ):
            raise ValueError(
                "Human animation physics_step must be a nonnegative integer."
            )
        first_binding = self._physics_step is None
        self._physics_step = int(physics_step)
        if first_binding:
            # A reset-time render may already have observed this graph with
            # no episode clock. Emit its real initial timed snapshot again.
            self._last_recorded_graph_state.clear()
            self.poll_animation_graph(observation_boundary="episode_clock_bound")

    def poll_animation_graph(self, *, observation_boundary: str) -> None:
        """Observe native clip families without advancing time or changing FSM.

        The explicit 60 Hz pump and dt=0 native renders can both evaluate graph
        conditions. Call immediately after either boundary, with the current
        physics tick already bound. A changed observation is not the exact
        continuous-time transition instant, nor a seat-contact certificate.
        """
        from tarsphere.infra.control.human_geometry import read_native_biped_graph_state

        for human_id, state in self._states.items():
            if (
                human_id in self._scene_bound_ids
                or state.body_status == "anim_people_pending"
            ):
                continue
            graph_path = str(state.metadata.get("animation_graph_path", ""))
            try:
                instance = (
                    self._fetch_character_script_instance(human_id)
                    if graph_path
                    else None
                )
                snapshot = read_native_biped_graph_state(
                    getattr(instance, "character", None), graph_path
                )
            except Exception as exc:
                # Diagnostics must preserve an unavailable measurement, never
                # block motion or replace it with a command-phase inference.
                snapshot = {
                    "state": "unknown",
                    "status": "unavailable",
                    "graph_path": graph_path,
                    "active_clip_families": [],
                    "active_clip_paths": [],
                    "nodes": {},
                    "errors": {"native_character": f"{type(exc).__name__}: {exc}"},
                }
            state.metadata["animation_graph_state"] = snapshot
            if self._last_recorded_graph_state.get(human_id) == snapshot:
                continue
            self._replay_log.append(
                {
                    "event": "human_animation_graph_state_changed",
                    "physics_step": self._physics_step,
                    "observation_boundary": observation_boundary,
                    "human_id": human_id,
                    "animation_graph_state": snapshot["state"],
                    "animation_graph_read_status": snapshot["status"],
                    "graph_path": snapshot["graph_path"],
                    "active_clip_families": list(snapshot["active_clip_families"]),
                    "active_clip_paths": list(snapshot["active_clip_paths"]),
                    "native_nodes": dict(snapshot["nodes"]),
                    "errors": dict(snapshot["errors"]),
                }
            )
            self._last_recorded_graph_state[human_id] = snapshot

    def get_replay_log(self) -> list[Dict[str, Any]]:
        return [dict(item) for item in self._replay_log]

    def poll_native_motion_trace(
        self, *, observation_boundary: str, force: bool = False
    ) -> None:
        """Opt-in 30 Hz active-action / 1 Hz ambient readback, without a pump.

        The environment supplies the real physics clock. Extra zero-time
        renders do not create duplicate samples. Unknown setup clocks stay
        unknown; no zero timestamp is fabricated.
        """
        if not self._capture_native_motion_trace or self._physics_step is None:
            return
        from tarsphere.infra.control.human_native_motion_trace import (
            measured,
            read_native_motion,
        )

        tick = self._physics_step
        for human_id, state in self._states.items():
            if (
                human_id in self._scene_bound_ids
                or state.body_status == "anim_people_pending"
            ):
                continue
            period = 8 if state.action_status == "ongoing" else 240
            previous = self._last_native_motion_trace_tick.get(human_id)
            if not force and previous is not None and tick - previous < period:
                continue
            snapshot = measured(
                lambda: read_native_motion(
                    self._fetch_character_script_instance(human_id)
                )
            )
            self._replay_log.append(
                {
                    "event": "human_native_motion_snapshot",
                    "physics_step": tick,
                    "human_id": human_id,
                    "observation_boundary": observation_boundary,
                    "body_state": state.to_observation(),
                    "sampling_period_ticks": period,
                    "native_readback": snapshot,
                }
            )
            self._last_native_motion_trace_tick[human_id] = tick

    def position(self, human_id: str) -> Optional[list[float]]:
        state = self.get_state(human_id)
        return list(state.position) if state is not None else None

    def yaw(self, human_id: str) -> float:
        state = self.get_state(human_id)
        return float(state.yaw_rad) if state is not None else 0.0

    def set_physical_status(
        self,
        human_id: str,
        status: str,
        *,
        error: str = "",
    ):
        state = self.get_state(human_id)
        if state is None:
            return
        state.physical_status = status
        state.last_error = error

    # ------------------------------------------------------------------
    # Action interface
    # ------------------------------------------------------------------

    def apply_action(self, human_id: str, action: Dict[str, Any]) -> HumanBodyResult:
        state = self._states.setdefault(
            human_id,
            HumanBodyState(human_id=human_id, position=[0.0, 0.0, 0.0]),
        )
        command = str(action.get("command") or "")
        if command not in HUMAN_RUNTIME_CAPABILITIES.command_names:
            return self._failed(
                human_id,
                command,
                f"Unsupported human body command: {command}.",
            )

        if command in self.SPEECH_COMMANDS:
            # Speech is synchronous but does not complete or cancel body work.
            # The same pose/motion state is consumed by the episode FSM.
            self._sync(human_id)
            return HumanBodyResult(
                human_id=human_id,
                command=command,
                executed=True,
                state=state,
            )

        if human_id in self._scene_bound_ids:
            if command in {"idle", "listen"}:
                state.animation_state = "native_static_pose"
                state.action_status = "idle"
                state.active_action = ""
                state.target_position = None
                self._sync(human_id)
                return HumanBodyResult(human_id, command, True, state=state)
            return self._failed(
                human_id,
                command,
                "A scene-bound static Human supports speech but not body motion.",
            )

        if command in {"idle", "listen"}:
            duration = self._action_duration(action, 5.0)
            self._inject_command(human_id, f"{human_id} Idle {duration}")
            state.animation_state = "idle"
            state.action_status = "idle"
            state.active_action = ""
            state.target_position = None
            self._goto_waypoint_queues.pop(human_id, None)
            self._sync(human_id)
            return HumanBodyResult(human_id, command, True, state=state)

        if command == "look_around":
            duration = self._action_duration(action, 5.0)
            self._inject_command(human_id, f"{human_id} LookAround {duration}")
            state.animation_state = "looking_around"
            state.action_status = "ongoing"
            state.active_action = command
            state.target_position = None
            self._sync(human_id)
            return HumanBodyResult(human_id, command, True, state=state)

        if command == "sit":
            seat_prim = str(
                action.get("seat_prim")
                or action.get("seat_prim_path")
                or action.get("target_prim")
                or ""
            ).strip()
            if not seat_prim:
                return self._failed(
                    human_id,
                    command,
                    "sit requires a seat_prim path (the seat USD prim).",
                )
            duration = self._action_duration(action, self.DEFAULT_SIT_DURATION_SEC)
            # Sit walks to <seat>/walk_to_offset before it will play the sit
            # animation.  GRScenes furniture ships without that child, and the
            # fallback (the seat's own origin, inside the mesh) is unreachable,
            # so the character orbits the seat and never sits.  Author the
            # offsets first; a seat with no reachable approach fails loudly
            # here rather than reporting 'sitting' while standing.
            try:
                seat_offsets = self._ensure_seat_interact_offsets(seat_prim)
            except Exception as exc:
                return self._failed(
                    human_id,
                    command,
                    f"sit cannot resolve an approach point for {seat_prim}: {exc}",
                )
            try:
                self._enable_sit_pose(human_id, state)
            except Exception as exc:
                return self._failed(
                    human_id, command, f"sit cannot enable its native pose: {exc}"
                )
            self._inject_command(human_id, f"{human_id} Sit {seat_prim} {duration}")
            state.target_position = None
            state.metadata["seat_prim"] = seat_prim
            state.metadata["seat_offsets"] = seat_offsets
            # The official Sit command includes walking to the seat, sitting,
            # holding the pose and standing up; injection is not a seated pose.
            state.animation_state = "sit_sequence"
            state.action_status = "ongoing"
            state.active_action = command
            self._sync(human_id)
            return HumanBodyResult(
                human_id,
                command,
                True,
                feedback=(
                    f"Human {human_id} sitting at {seat_prim} for {duration}s via "
                    "omni.anim.people Sit. The seat prim should expose walk_to_offset "
                    "and interact_offset child xforms for correct alignment."
                ),
                state=state,
            )

        if command == "queue":
            queue_name = str(
                action.get("queue_name") or action.get("name") or ""
            ).strip()
            spots = action.get("spots")
            if not queue_name:
                return self._failed(human_id, command, "queue requires a queue_name.")
            if not spots:
                return self._failed(
                    human_id,
                    command,
                    "queue requires a non-empty spots list of [x, y, z, rotation].",
                )
            lines = [f"Queue {queue_name}"]
            for index, spot in enumerate(spots):
                values = [float(item) for item in list(spot)[:4]]
                if len(values) < 4:
                    return self._failed(
                        human_id,
                        command,
                        f"each queue spot needs [x, y, z, rotation]; got {spot}.",
                    )
                x_coord, y_coord, z_coord, rotation = values
                lines.append(
                    f"Queue_Spot {queue_name} {index} "
                    f"{x_coord} {y_coord} {z_coord} {rotation}"
                )
            lines.append(f"{human_id} Queue {queue_name}")
            first_action = str(action.get("first_action") or "").strip()
            if first_action:
                lines.append(f"{human_id} {first_action}")
            leave_target = action.get("leave_target")
            if leave_target is not None and len(leave_target) >= 2:
                leave = [float(item) for item in list(leave_target)[:3]]
                leave_rotation = action.get("leave_rot", "_")
                lines.append(
                    f"{human_id} Dequeue {queue_name} "
                    f"{leave[0]} {leave[1]} {leave[2]} {leave_rotation}"
                )
            self._inject_command(human_id, lines)
            state.metadata["queue_name"] = queue_name
            state.target_position = None
            state.animation_state = "queueing"
            state.action_status = "ongoing"
            state.active_action = command
            self._sync(human_id)
            return HumanBodyResult(
                human_id,
                command,
                True,
                feedback=(
                    f"Human {human_id} entered queue {queue_name} "
                    f"({len(spots)} spots) via omni.anim.people."
                ),
                state=state,
            )

        ira_command = self.CUSTOM_ANIMATED_COMMANDS.get(command.lower())
        if ira_command is not None:
            duration = self._action_duration(action, self.DEFAULT_CUSTOM_DURATION_SEC)
            self._inject_command(human_id, f"{human_id} {ira_command} {duration}")
            state.target_position = None
            state.metadata["ira_command"] = ira_command
            state.animation_state = command.lower()
            state.action_status = "ongoing"
            state.active_action = command
            self._sync(human_id)
            return HumanBodyResult(
                human_id,
                command,
                True,
                feedback=(
                    f"Human {human_id} performing {ira_command} for {duration}s via "
                    "omni.anim.people custom command."
                ),
                state=state,
            )

        if command == "goto":
            target = action.get("target")
            if not self._valid_target(target):
                return self._failed(
                    human_id, command, f"{command} requires a target position."
                )
            final_target = [float(value) for value in target[:3]]
            end_rot = action.get("end_rot", "_")
            if end_rot != "_":
                try:
                    end_rot = float(end_rot)
                    if not math.isfinite(end_rot):
                        raise ValueError("nonfinite orientation")
                except (ValueError, TypeError):
                    return self._failed(
                        human_id, command, "goto end_rot must be finite degrees or _."
                    )
                if (
                    not state.pose_stale
                    and math.dist(state.position, final_target)
                    <= self.GOTO_COMMAND_DEDUP_TOLERANCE_M
                ):
                    if (
                        yaw_error_rad(state.yaw_rad, math.radians(end_rot))
                        > SEAT_YAW_TOLERANCE_RAD
                    ):
                        return self._failed(
                            human_id,
                            command,
                            "GoTo cannot establish a different yaw through a zero-length path; "
                            "provide a nonzero native approach route.",
                        )
                    end_rot = "_"
            route = action.get("waypoints")
            if route is None:
                route_points = [final_target]
            else:
                route_points = [
                    [float(value) for value in point[:3]]
                    for point in route
                    if self._valid_target(point)
                ]
                if len(route_points) != len(route):
                    return self._failed(
                        human_id,
                        command,
                        "goto waypoints must all be finite world XYZ positions.",
                    )
                if not route_points or math.dist(route_points[-1], final_target) > 1e-6:
                    return self._failed(
                        human_id,
                        command,
                        "goto waypoints must end at the requested target.",
                    )
            # A close position does not satisfy a requested final orientation.
            # Keep the native GoTo alive so the SDK owns the turn; do not set
            # the character transform ourselves.
            if not state.pose_stale and end_rot == "_":
                route_points = [
                    point
                    for point in route_points
                    if math.dist(point[:2], state.position[:2])
                    > HUMAN_GOTO_ARRIVAL_TOLERANCE_M
                ]
            if not route_points:
                if (
                    state.active_action == "goto"
                    and state.action_status == "ongoing"
                    and self._goto_already_injected(human_id, final_target)
                ):
                    self._sync(human_id)
                    return HumanBodyResult(
                        human_id,
                        command,
                        True,
                        feedback="GoTo is near its target and still finishing the native command.",
                        state=state,
                    )
                state.animation_state = "idle"
                state.action_status = "succeed"
                state.active_action = command
                state.target_position = None
                self._goto_waypoint_queues.pop(human_id, None)
                self._sync(human_id)
                return HumanBodyResult(
                    human_id,
                    command,
                    True,
                    feedback=f"Human {human_id} is already at the GoTo target.",
                    state=state,
                )
            target_position = route_points[0]
            self._goto_waypoint_queues[human_id] = route_points[1:]
            state.metadata["route_world_xyz"] = [list(point) for point in route_points]
            # Suppress redundant GoTo.  omni.anim.people keeps executing a GoTo
            # until the character arrives, so a stateless human policy that
            # re-decides "move to X" every step would re-inject the same GoTo
            # each step.  That grows the command queue and restarts path
            # planning on an already-reached / trivial path, producing
            # degenerate BezierSplines that crash motion-matching
            # (BezierSpline::NearestPointOnCurve).  Inject once; subsequent
            # same-target or already-arrived GoTos are no-ops.
            if end_rot == "_" and self._goto_is_redundant(state, target_position):
                self._sync(human_id)
                return HumanBodyResult(
                    human_id,
                    command,
                    True,
                    feedback=(
                        f"Human {human_id} GoTo suppressed — already at / walking"
                        f" to {target_position[:3]}."
                    ),
                    state=state,
                )
            # Injection-level dedupe (independent of pose read-back, which is
            # unreliable mid-walk): if an equivalent GoTo is already in flight,
            # do not re-inject.  Re-injecting a destination the character has
            # reached re-plans a zero-length path that crashes omni.anim.people
            # motion-matching (NavigationPath::Advance) on a recycled runtime.
            if self._goto_already_injected(human_id, target_position):
                self._sync(human_id)
                return HumanBodyResult(
                    human_id,
                    command,
                    True,
                    feedback=(
                        f"Human {human_id} GoTo suppressed — same destination as"
                        f" the in-flight GoTo; not re-injected."
                    ),
                    state=state,
                )
            self._inject_command(
                human_id,
                f"{human_id} GoTo {target_position[0]} {target_position[1]} {target_position[2]} {end_rot}",
            )
            self._last_injected_goto[human_id] = (
                target_position[0],
                target_position[1],
            )
            state.target_position = target_position
            state.animation_state = "goto_pending"
            state.action_status = "ongoing"
            state.active_action = command
            # Admission may interrupt Sit, whose native stand-up must finish
            # before GoTo begins. Only native observations establish the pose.
            self._poll_command_completion(human_id, state)
            self._sync(human_id)
            return HumanBodyResult(
                human_id,
                command,
                True,
                feedback=(
                    f"Human {human_id} accepted an ordered {len(route_points)}-leg "
                    f"route toward {[round(v, 3) for v in final_target]} via "
                    "omni.anim.people."
                ),
                state=state,
            )

        return self._failed(
            human_id, command, f"Unsupported human body command: {command}."
        )

    def step(self, dt: float):
        """Advance body state by reading back character positions from the stage.

        Animation and locomotion are driven by ``omni.anim.people`` internally.
        This method reads live transforms and command outcomes, then queues
        the next route leg or ambient command. It never advances the clock.

        Polling alone does not move anybody.  ``omni.anim.people`` is clocked by
        Kit app updates -- ``character_behavior.on_update`` is an app-update
        callback -- and ``SimulationContext.step(render=False)`` never calls
        ``app.update()``.  When the room observer is off (every evaluation run)
        a decision window therefore delivers one Kit update in total and the
        characters are effectively frozen.  :meth:`pump_motion_window` is the
        opt-in remedy; see its docstring for why it is not called from here.
        """
        for human_id, state in self._states.items():
            if human_id in self._scene_bound_ids:
                self._sync(human_id)
                continue
            self._poll_character_position(human_id, state)
            self._poll_command_completion(human_id, state)
            self._sync(human_id)

    def _start_ambient_look_around(self, human_id: str, state: HumanBodyState) -> None:
        self._inject_command(
            human_id,
            f"{human_id} LookAround {self.AMBIENT_LOOK_AROUND_DURATION_SEC}",
        )
        state.animation_state = "looking_around"
        state.active_action = ""
        state.target_position = None

    def _poll_command_completion(self, human_id: str, state: HumanBodyState) -> None:
        """Use IRA's command-end event, not wall time, to finish an animation.

        A queued command has not finished even when no current command exists
        yet. Sit may spend an unbounded time approaching its chair before its
        duration starts, and its stand-up transition must finish too.
        """
        instance = self._fetch_character_script_instance(human_id)
        if instance is None:
            raise RuntimeError(
                f"Cannot read anim_people command state for {human_id!r}."
            )
        current = getattr(instance, "current_command", None)
        pending = list(getattr(instance, "commands", []) or [])
        if current is not None:
            native_command = current.get_command_name()
            if native_command == "Sit":
                phase = getattr(current, "current_action", None)
                state.animation_state = {
                    "walk": "walking_to_seat",
                    "sit": "sitting",
                    "stand": "standing_up",
                }.get(phase, "sit_sequence")
                # These are native phase measurements, not a seat-contact
                # certificate; the demo probe records geometry separately.
                state.metadata["sit_phase"] = phase
                state.metadata["sit_time_s"] = float(getattr(current, "sit_time", 0.0))
                state.metadata["sit_position_blend"] = float(
                    getattr(current, "_char_lerp_t", 0.0)
                )
            else:
                state.animation_state = {
                    "GoTo": "walking",
                    "LookAround": "looking_around",
                    "Idle": "idle",
                    "QueueCmd": "queueing",
                    "Dequeue": "walking",
                    "TypeKeyboard": "type_keyboard",
                    "PushButton": "push_button",
                }.get(native_command, str(native_command).lower())
        if state.action_status == "fail":
            return
        if self._pending_command_ids.get(human_id):
            if current is None and not pending:
                raise RuntimeError(
                    f"anim_people discarded {human_id!r} commands without a completion event."
                )
            return
        if current is not None or pending:
            return
        if state.action_status == "ongoing":
            # GoTo completion is owned by measured route arrival, including
            # all ordered waypoints; an empty queue cannot substitute for it.
            if state.active_action == "goto" or state.pose_stale:
                return
            state.action_status = "succeed"
        if state.action_status in {"idle", "succeed"}:
            self._start_ambient_look_around(human_id, state)

    # ------------------------------------------------------------------
    # Motion window
    # ------------------------------------------------------------------

    def has_active_motion(self) -> bool:
        """Whether any character currently holds an unfinished command.

        Ambient LookAround is deliberately excluded: it must not block an
        authored action or make the event FSM believe work is unfinished.
        """
        for state in self._states.values():
            if getattr(state, "body_status", "") == "anim_people_pending":
                continue
            if getattr(state, "action_status", "") == "ongoing":
                return True
        return False

    def active_motion_ids(self) -> list[str]:
        """Ids of the characters holding an unfinished command."""
        return [
            human_id
            for human_id, state in self._states.items()
            if getattr(state, "body_status", "") != "anim_people_pending"
            and getattr(state, "action_status", "") == "ongoing"
        ]

    def pump_motion_window(self, *, physics_tick: int, pump_every: int) -> int:
        """Advance the animation clock on a Human-activity gate.

        Kept out of :meth:`step` deliberately: reading state or queuing the
        next command must not advance time. The environment owns the cadence
        and enables this clock only for the animation backend.

        Returns the number of Kit updates issued, so callers can account for
        them. Animated actors also need this clock while looking around.

        Note the pump *cadence* and the Kit *step size* are independent knobs
        and must be set together.  A pump declares ``manual_step_size`` seconds
        of elapsed time regardless of how many physics ticks actually passed, so
        pumping every ``N`` ticks while the step size stays at ``1/240``
        under-reports elapsed time by ``N`` times and the characters crawl at
        ``1/N`` speed.  For animation time to track simulated time::

            manual_step_size = pump_every / SIMULATION_PHYSICS_HZ

        Measured 2026-08-19 (ego-only, no observer): at ``pump_every=4`` with a
        matching ``1/60`` step size the character walks 1.03 m/s against a
        nominal 1.2 m/s, versus 0.14 m/s when the step size is left at 1/240.
        """
        if pump_every < 1:
            raise ValueError(f"pump_every must be >= 1, got {pump_every}")
        if physics_tick % int(pump_every) != 0:
            return 0
        if not any(
            human_id not in self._scene_bound_ids
            and state.body_status != "anim_people_pending"
            for human_id, state in self._states.items()
        ):
            return 0
        self._pump_app_updates(count=1)
        self.poll_animation_graph(observation_boundary="after_human_pump")
        return 1

    # ------------------------------------------------------------------
    # omni.anim.people integration
    # ------------------------------------------------------------------

    def _ensure_animation_clips_loaded(self, biped_prim: Any):
        """Load local SkelAnimation clips as payloads under the biped prim.

        Biped_Setup.usd contains payload references to SkelAnimation clips that
        may not resolve from a local filesystem path.  This method checks whether
        the expected animation prims exist under the biped, and for each missing
        prim it creates a payload reference to the locally downloaded clip file
        in ``<characters_root>/Animations/``, matching the relative payloads
        authored in ``Biped_Setup.usd``.
        """
        if biped_prim is None or not biped_prim.IsValid():
            return

        try:
            import omni.kit.commands
            import omni.usd
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        anim_base = str(biped_prim.GetPath()) + "/CharacterAnimation/Animation"
        anim_dir = Path(self._character_assets_path).parent / "Animations"
        if not anim_dir.is_dir():
            raise FileNotFoundError(
                f"Isaac People Animations directory is missing: {anim_dir}"
            )

        required_clips = HUMAN_RUNTIME_CAPABILITIES.required_animation_clips
        missing = [
            anim_dir / f"{clip_name}.skelanim.usd"
            for clip_name in required_clips
            if not (anim_dir / f"{clip_name}.skelanim.usd").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Required Isaac People animation clips are missing: "
                + ", ".join(str(path) for path in missing)
            )

        loaded = 0
        skipped = 0
        for clip_name in required_clips:
            prim_path = f"{anim_base}/{clip_name}_skelanim"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                skipped += 1
                continue

            clip_file = anim_dir / f"{clip_name}.skelanim.usd"
            try:
                omni.kit.commands.execute(
                    "CreatePayload",
                    path_to=prim_path,
                    asset_path=str(clip_file),
                    usd_context=omni.usd.get_context(),
                    select_prim=False,
                )
                loaded += 1
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load required Human animation payload "
                    f"{clip_name!r} from {clip_file}: {exc}"
                ) from exc

        if loaded > 0:
            logger.info(
                f"[AnimPeopleBodyExecutor] Loaded {loaded} local animation clips "
                f"as payloads under {anim_base} ({skipped} already present)."
            )
            self._pump_app_updates(count=1)

    def _register_custom_commands(self, biped_prim: Any):
        """Best-effort: stamp IRA custom-command attributes on loaded clips.

        IRA custom commands (e.g. ``TypeKeyboard``, ``PushButton``) require each
        SkelAnimation clip prim to carry ``CustomCommandName`` and
        ``CustomCommandTemplate`` USD attributes, and the clip to be listed in
        IRA's ``custom_command_tracking.json`` so the anim graph node is created
        at Set Up Simulation time.  This helper sets the USD attributes on the
        clips that are already loaded as payloads under the biped; it does not
        modify IRA extension data.  Command injection still fails loud (see
        ``_inject_command``) if the running anim graph does not recognize the
        command, so an unregistered custom command is never a silent no-op.
        """
        if biped_prim is None or not biped_prim.IsValid():
            return
        try:
            import omni.usd
            from pxr import Sdf  # noqa: F401
        except Exception:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        anim_base = str(biped_prim.GetPath()) + "/CharacterAnimation/Animation"
        registrations = {
            f"{command}_skelanim": (backend_command, "Timing")
            for command, backend_command in self.CUSTOM_ANIMATED_COMMANDS.items()
        }
        stamped = 0
        for clip_prim_name, (command_name, template) in registrations.items():
            prim = stage.GetPrimAtPath(f"{anim_base}/{clip_prim_name}")
            if not prim.IsValid():
                continue
            self._ensure_custom_command_attrs(prim, command_name, template)
            stamped += 1
        if stamped:
            logger.info(
                "[AnimPeopleBodyExecutor] Stamped IRA custom-command attributes "
                f"on {stamped} clip(s) under {anim_base}."
            )

    @staticmethod
    def _ensure_custom_command_attrs(prim: Any, command_name: str, template: str):
        from pxr import Sdf

        for attr_name, value in (
            ("CustomCommandName", command_name),
            ("CustomCommandTemplate", template),
        ):
            attr = prim.GetAttribute(attr_name)
            if attr is None or not attr.IsValid():
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.String)
            if attr.Get() != value:
                attr.Set(value)

    def _initialize_people_api(self):
        """Attempt to bind to the omni.anim.people extension."""
        if self._initialized:
            return

        try:
            import omni.usd
            import omni.anim.people
            from omni.anim.people import get_instance
        except ImportError as exc:
            raise RuntimeError(
                "omni.anim.people extension is not available.  "
                "Ensure Isaac Sim is running with the omni.anim.people "
                f"extension enabled.  Import error: {exc}"
            ) from exc

        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError(
                "USD stage is not available.  Cannot initialize AnimPeopleBodyExecutor."
            )

        self._people_api = get_instance()
        if self._people_api is None:
            raise RuntimeError(
                "omni.anim.people extension is loaded but the API instance is None.  "
                "The extension may not have started correctly."
            )

        logger.info("[AnimPeopleBodyExecutor] omni.anim.people API initialized.")
        self._initialized = True

    @staticmethod
    def _stage_identity() -> Optional[str]:
        """Name the context's stage in a way that survives a stage attach.

        ``UsdContext.get_stage_id()`` and the ``UsdUtils`` stage-cache id are
        both *re-issued* by ``attach_stage_with_callback`` -- measured 9223001 ->
        9223002 for one stage that ``is`` itself before and after.  Keying the
        one-shot guard on either therefore re-arms it, and the second attach
        lands on top of the live Replicator render products: every frame is then
        discarded (``OgnSdOnNewFrame ... discarded``) and the ego camera never
        produces an image again.  The root layer identifier names the stage
        itself and does not move.
        """
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        return str(stage.GetRootLayer().identifier)

    def author_character_animation_graph(self):
        """Author ``Biped_Setup`` so the next stage attach can build the manager.

        ``omni.anim.graph.core`` builds its CharacterManager from the
        ``AnimationGraph`` stage-update node's *attach* callback, and only when
        an AnimationGraph prim is already on the stage.  TARSphere used to author
        ``Biped_Setup`` while spawning a character, which is long after
        TARSphere opened and attached the scene, so the manager never
        initialized at all::

            [omni.anim.graph.core.plugin] CharacterManager::Shutdown() called
            without a prior successful call to CharacterManager::Initialize().

        With no manager, ``ag.get_character()`` returns None for every prim --
        the C++ side then blames the prim ("is not a SkelRoot prim or does not
        apply AnimationGraphAPI") even though USD *and* Fabric both report a
        SkelRoot carrying a correctly targeted AnimationGraphAPI.  Every
        character's BehaviorScript then refuses to initialize and the rollout
        aborts before its first frame.  Ticking does not help: measured 1200
        updates on a playing timeline, still None.

        This only authors USD.  The attach itself belongs to
        :class:`IsaacAutonomousEnv`, which owns the recovery it requires --
        see :meth:`IsaacAutonomousEnv._prepare_anim_people_character_runtime`.
        """
        try:
            from isaacsim.replicator.agent.core.stage_util import CharacterUtil
        except Exception as exc:
            raise RuntimeError(
                "isaacsim.replicator.agent.core is required to author the "
                f"anim_people AnimationGraph: {exc}"
            ) from exc

        biped_prim = CharacterUtil.load_default_biped_to_stage()
        if biped_prim is None or not biped_prim.IsValid():
            raise RuntimeError(
                "Could not author the anim_people Biped_Setup prim; without its "
                "AnimationGraph no character can ever register."
            )
        self._ensure_animation_clips_loaded(biped_prim)
        self._register_custom_commands(biped_prim)
        self._pump_app_updates(count=5)
        return biped_prim

    def mark_character_runtime_prepared(self):
        """Record that a stage attach has seen the AnimationGraph."""
        self._prepared_stage_id = self._stage_identity()
        self._character_runtime_prepared = True

    def require_character_runtime_prepared(self):
        """Require stage preparation before spawning a character.

        Attaching a stage under live render products invalidates ``World`` and
        orphans the Replicator graph. Prepare the AnimationGraph during reset,
        before cameras and characters are created.
        """
        stage_key = self._stage_identity()
        if self._character_runtime_prepared and stage_key == self._prepared_stage_id:
            return
        raise RuntimeError(
            "The omni.anim.graph.core CharacterManager was never prepared for "
            f"stage {stage_key!r} (prepared={self._prepared_stage_id!r}).  "
            "reset() must author the AnimationGraph and attach the stage -- "
            "with the TARSphere runtime refresh that an attach requires -- "
            "before any character is spawned."
        )

    def _spawn_character(
        self,
        human_id: str,
        position: list[float],
        *,
        character_name: Optional[str] = None,
        start_yaw_rad: float = 0.0,
    ):
        """Spawn and configure a character using the official IRA utilities."""
        try:
            import omni.kit.commands
            import omni.usd
            from isaacsim.replicator.agent.core.settings import (
                BehaviorScriptPaths,
                PrimPaths,
            )
            from isaacsim.replicator.agent.core.stage_util import CharacterUtil
            from pxr import UsdGeom

            # Top-level `import NavSchema` avoids the deprecated pxr shim.
            import NavSchema  # noqa: E402
        except Exception as exc:
            raise RuntimeError(
                "isaacsim.replicator.agent.core is required to spawn anim_people "
                f"characters. Import/setup error: {exc}"
            ) from exc

        if not math.isfinite(start_yaw_rad):
            raise ValueError("Initial human heading must be finite.")
        # IRA CharacterUtil accepts world Z rotation in degrees and authors
        # it while loading the root, before graph/BehaviorScript attachment.
        # Episode yaw is a heading; explicit nonzero headings need the same
        # +90-degree character-axis offset as native locomotion. Keep the
        # historical default0 spawn orientation for unspecified/zero headings.
        spawn_rotation_deg = (
            (math.degrees(start_yaw_rad) + self.CHARACTER_FACING_YAW_OFFSET_DEG)
            if start_yaw_rad != 0.0
            else 0.0
        )
        asset_name = self._resolve_character_name(character_name)
        character_usd_path = self._resolve_character_usd(asset_name)
        # The AnimationGraph and its stage attach must already be in place: by
        # now TARSphere's render products are live and attaching would kill
        # them.  This only re-checks the contract; reset() does the work.
        self.require_character_runtime_prepared()
        CharacterUtil.delete_character_prim(human_id)
        # The first child below Characters remains the public Human name used
        # by IRA. Only its referenced asset/SkelRoot receives a fresh identity.
        stage = omni.usd.get_context().get_stage()
        actor_path = f"{PrimPaths.characters_parent_path()}/{human_id}"
        UsdGeom.Xform.Define(stage, actor_path)
        asset_stage_name = f"{human_id}/spawn_{self._spawn_epoch:06d}"
        root_prim = CharacterUtil.load_character_usd_to_stage(
            str(character_usd_path),
            position,
            spawn_rotation_deg,
            asset_stage_name,
        )
        if root_prim is None or not root_prim.IsValid():
            raise RuntimeError(
                f'Failed to spawn anim_people character "{human_id}" from {character_usd_path}.'
            )
        # Give every runtime Human one stable semantic instance label.  The
        # ego-camera tight-bbox annotator inherits this label across the
        # character meshes, allowing the private Golden oracle to emit a real
        # visual box instead of projecting a loose collision cylinder.
        try:
            from isaacsim.core.utils.semantics import add_update_semantics

            add_update_semantics(root_prim, human_id, "class")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to assign semantic id {human_id!r} to AnimPeople root "
                f"{root_prim.GetPath()}: {exc}"
            ) from exc

        biped_prim = CharacterUtil.load_default_biped_to_stage()
        # Ensure animation clips are loaded as payloads under the biped prim.
        # Biped_Setup.usd contains payload references to SkelAnimation clips, but
        # those references may not resolve when the USD is loaded from a local path.
        # We explicitly load the local clips from the sibling Animations
        # directory required by Biped_Setup.usd.
        self._ensure_animation_clips_loaded(biped_prim)
        self._register_custom_commands(biped_prim)
        anim_graph = CharacterUtil.get_anim_graph_from_character(biped_prim)
        skelroot = CharacterUtil.get_character_skelroot_by_root(root_prim)
        if skelroot is None or not skelroot.IsValid():
            raise RuntimeError(
                f'AnimPeople character "{human_id}" has no SkelRoot under {root_prim.GetPath()}.'
            )
        CharacterUtil.setup_animation_graph_to_character([skelroot], anim_graph)
        CharacterUtil.setup_python_scripts_to_character(
            [skelroot],
            BehaviorScriptPaths.behavior_script_path(),
        )
        try:
            omni.kit.commands.execute(
                "ApplyNavMeshAPICommand",
                prim_path=str(root_prim.GetPath()),
                api=NavSchema.NavMeshExcludeAPI,
            )
        except Exception as exc:
            logger.warning(
                f"[AnimPeopleBodyExecutor] Could not apply NavMeshExcludeAPI to "
                f"{root_prim.GetPath()}: {exc}"
            )

        state = self._states[human_id]
        state.body_status = "anim_people_usd_avatar"
        state.physical_status = "navmesh_enabled"
        state.metadata["character_asset"] = asset_name
        state.metadata["spawn_epoch"] = self._spawn_epoch
        state.metadata["initial_heading_rad"] = start_yaw_rad
        state.metadata["native_spawn_rotation_deg"] = spawn_rotation_deg
        state.metadata["animation_graph_path"] = str(anim_graph.GetPath())
        self._character_prim_paths[human_id] = str(root_prim.GetPath())
        self._character_skelroot_paths[human_id] = str(skelroot.GetPath())
        self._pump_app_updates()
        logger.info(
            f'[AnimPeopleBodyExecutor] Spawned "{human_id}" from "{asset_name}" '
            f"at {position[:3]} under {root_prim.GetPath()}."
        )

    def _bind_scene_person(self, human: Any, source_path: str) -> None:
        """Bind an exact GRScene person without spawning or animating it."""
        import omni.usd
        from isaacsim.core.utils.semantics import add_update_semantics
        from tarsphere.infra.control.scene_person_geometry import (
            tight_visible_mesh_bounds,
        )

        runtime_path = source_path.replace("/Root", "/World/env_0/scene", 1)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Scene-bound Human requires a live USD stage.")
        prim = stage.GetPrimAtPath(runtime_path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(
                f"Scene-bound Human {human.id!r} prim is missing: {runtime_path}"
            )
        actual_min, actual_max = tight_visible_mesh_bounds(prim)
        expected_min = np.asarray(human.scene_person_bounds_min_m, dtype=float)
        expected_max = np.asarray(human.scene_person_bounds_max_m, dtype=float)
        error_m = float(
            max(
                np.max(np.abs(actual_min - expected_min)),
                np.max(np.abs(actual_max - expected_max)),
            )
        )
        if error_m > 0.03:
            raise RuntimeError(
                f"Scene-bound Human {human.id!r} geometry is stale: "
                f"bounds error {error_m:.3f} m at {runtime_path}; "
                f"expected_min={expected_min.tolist()} "
                f"expected_max={expected_max.tolist()} "
                f"actual_min={actual_min.tolist()} actual_max={actual_max.tolist()}"
            )
        add_update_semantics(prim, str(human.id), "class")
        human_id = str(human.id)
        self._scene_bound_ids.add(human_id)
        self._character_prim_paths[human_id] = runtime_path
        state = self._states[human_id]
        state.body_status = "scene_bound_static_person"
        state.physical_status = "collision_proxy_pending"
        state.metadata["pose_source"] = "authored_scene_person_prim"
        state.metadata["scene_person_prim_path"] = source_path

    def _initialize_character_scripts(self, human_ids: list[str]):
        """Wait for official BehaviorScript instances and initialize runtime state.

        Isaac Replicator Agent command injection is mediated by the Python
        Scripting Component.  Attaching the script USD attribute is not enough:
        the script manager must instantiate the BehaviorScript, the character
        must initialize its animation/nav managers, and the AgentManager must
        register the instance before command injection is reliable.
        """
        if not human_ids:
            return
        self._enter_character_runtime()
        for human_id in human_ids:
            self._initialize_character_script(human_id)

    def _initialize_character_script(
        self,
        human_id: str,
        *,
        timeout_seconds: float = 10.0,
        min_updates: int = 60,
        max_seconds: float = 600.0,
    ):
        """Drive app updates until the character's BehaviorScript is live.

        What this wait actually needs is *update ticks*: the script manager, the
        character's animation graph and the AgentManager each register on a
        later tick than the one that attached them. A pure wall-clock budget
        conflates "enough ticks" with "enough seconds", and that breaks as soon
        as one tick is expensive -- the first update after a scene load includes
        the initial Hydra render, measured here at 14 minutes for a GRScene home
        scene with 1238 static bodies. The 10s budget was then spent inside a
        single tick, so ``init_character`` was asked once, answered False
        because the anim-graph character was not registered yet, and the wait
        gave up having never pumped a second time.

        So the budget is whichever allowance is more generous, seconds or
        ticks, under an absolute ceiling so a genuinely dead script still fails
        rather than hanging the run.
        """
        try:
            from isaacsim.replicator.agent.core.agent_manager import AgentManager
        except Exception as exc:
            raise RuntimeError(
                "isaacsim.replicator.agent.core AgentManager is required for "
                f"anim_people command injection: {exc}"
            ) from exc

        started = time.monotonic()
        deadline = started + timeout_seconds
        hard_deadline = started + max_seconds
        updates = 0

        def pump(count: int = 1) -> None:
            nonlocal updates
            self._pump_app_updates(count=count)
            updates += count

        last_error = ""
        agent_manager = AgentManager.get_instance()
        on_play_called = False
        while time.monotonic() < hard_deadline and (
            updates < min_updates or time.monotonic() < deadline
        ):
            instance = self._fetch_character_script_instance(human_id)
            if instance is None:
                last_error = "BehaviorScript instance is not registered yet."
                pump()
                continue

            if not on_play_called and callable(getattr(instance, "on_play", None)):
                try:
                    instance.on_play()
                    on_play_called = True
                except Exception as exc:
                    last_error = f"on_play failed: {exc}"
                    pump()
                    continue
                pump(2)

            init_character = getattr(instance, "init_character", None)
            if callable(init_character):
                try:
                    if init_character() is False:
                        last_error = "init_character returned False."
                        pump()
                        continue
                except Exception as exc:
                    last_error = f"init_character failed: {exc}"
                    pump()
                    continue

            try:
                agent_manager.register_agent(human_id, getattr(instance, "prim_path"))
            except Exception as exc:
                last_error = f"AgentManager.register_agent failed: {exc}"
                pump()
                continue
            if agent_manager.get_agent_script_instance_by_name(human_id) is None:
                last_error = "AgentManager registration produced no script instance."
                pump()
                continue

            logger.info(
                f"[AnimPeopleBodyExecutor] Initialized BehaviorScript for "
                f'"{human_id}" at {getattr(instance, "prim_path", "<unknown>")} '
                f"after {updates} update(s), {time.monotonic() - started:.1f}s."
            )
            return

        skelroot_path = self._character_skelroot_paths.get(human_id, "<unknown>")
        raise RuntimeError(
            f'Could not initialize anim_people BehaviorScript for "{human_id}" '
            f"at {skelroot_path} after {updates} update(s) and "
            f"{time.monotonic() - started:.1f}s. {last_error}"
        )

    def _enter_character_runtime(self):
        """Ensure the timeline is running after scripts/AnimGraphs are attached.

        ``omni.kit.scripting`` broadcasts ``on_play`` only on a timeline PLAY
        event.  TARSphere spawns people after the main TARSphere reset has already
        started the timeline for NavMesh bake, so the official executor avoids
        stopping a live timeline here.  Per-character ``on_play`` is invoked
        once in ``_initialize_character_script`` for newly attached scripts.
        """
        try:
            import omni.timeline
        except Exception as exc:
            raise RuntimeError(
                "omni.timeline is required to initialize anim_people characters."
            ) from exc

        timeline = omni.timeline.get_timeline_interface()
        if timeline is None:
            raise RuntimeError(
                "Isaac timeline is unavailable; cannot initialize anim_people characters."
            )

        try:
            if not timeline.is_playing():
                timeline.play()
            self._pump_app_updates(count=8)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to enter anim_people runtime after spawning characters: {exc}"
            ) from exc

    def _inject_command(self, human_id: str, command):
        """Inject command line(s) through the official omni.anim.people API.

        ``command`` is either a single IRA command string or an iterable of
        strings injected as one atomic batch.  A multi-line batch is used by the
        ``Queue`` command, which first defines the queue (``Queue`` and
        ``Queue_Spot`` lines carry no agent prefix) and then makes the character
        enter/act/dequeue (``{id} ...`` lines).

        Command format follows the official IRA convention::

            Character_01 GoTo 10 10 0 90
            Character_01 Idle 5
            Character_01 LookAround 10
            Character_01 Sit /World/Chair 5
        """
        lines = (
            [command] if isinstance(command, str) else [str(line) for line in command]
        )
        try:
            from isaacsim.replicator.agent.core.agent_manager import AgentManager

            agent_manager = AgentManager.get_instance()
            instance = agent_manager.get_agent_script_instance_by_name(human_id)
            if instance is None:
                self._initialize_character_script(human_id)
                instance = agent_manager.get_agent_script_instance_by_name(human_id)
            if instance is None:
                raise RuntimeError(
                    f'No registered anim_people agent named "{human_id}".'
                )
            self._ensure_command_end_subscription()
            self._command_serial += 1
            identified_lines = [
                (f"hca:{human_id}:{self._command_serial}:{index}", line)
                for index, line in enumerate(lines)
            ]
            self._pending_command_ids[human_id] = {
                command_id
                for command_id, line in identified_lines
                if line.split()[:1] == [human_id]
            }
            self._issued_command_ids.update(self._pending_command_ids[human_id])
            state = self._states[human_id]
            state.last_error = ""
            state.metadata.pop("anim_people_command_result", None)
            agent_manager.inject_command(
                human_id,
                identified_lines,
                force_inject=True,
                instant=True,
            )
            expected_command = self._last_agent_command_name(lines, human_id)
            if expected_command and not self._script_queue_contains_command(
                instance, expected_command
            ):
                raise RuntimeError(
                    f'AgentManager did not queue "{expected_command}" for "{human_id}".'
                )
            logger.debug(f"[AnimPeopleBodyExecutor] Injected: {lines}")
        except Exception as exc:
            state = self._states.get(human_id)
            if state is not None:
                state.last_error = f"Command injection failed: {exc}"
                state.action_status = "fail"
            raise RuntimeError(
                f"Failed to inject anim_people command(s) {lines}: {exc}"
            ) from exc

    def _ensure_command_end_subscription(self) -> None:
        if self._command_end_subscription is not None:
            return
        import carb.eventdispatcher
        from omni.anim.people.settings import AgentEvent

        self._command_end_subscription = (
            carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=AgentEvent.CommandEndEvent,
                on_event=self._on_command_end,
                observer_name=f"hca/people/command_end/{id(self)}",
            )
        )
        self._command_start_subscription = (
            carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=AgentEvent.CommandStartEvent,
                on_event=self._on_command_start,
                observer_name=f"hca/people/command_start/{id(self)}",
            )
        )

    def _on_command_start(self, event: Any) -> None:
        payload = event.payload
        human_id = str(payload["agent_name"])
        if str(payload["command_id"]) not in self._issued_command_ids:
            return
        self._record_native_command_event("human_body_command_started", payload)

    def _record_native_command_event(self, event: str, payload: Dict[str, Any]) -> None:
        self._replay_log.append(
            {
                "event": event,
                "physics_step": self._physics_step,
                "human_id": str(payload["agent_name"]),
                "command_id": str(payload["command_id"]),
                "command": str(payload["command_name"]),
                "status": str(payload["status"]),
                "animation_time_code": float(payload["time_code"]),
            }
        )

    def _on_command_end(self, event: Any) -> None:
        """Reject failed/interrupted command ends and unrelated old batches."""
        payload = event.payload
        human_id = str(payload["agent_name"])
        command_id = str(payload["command_id"])
        if command_id not in self._issued_command_ids:
            return
        self._issued_command_ids.remove(command_id)
        self._record_native_command_event("human_body_command_ended", payload)
        pending = self._pending_command_ids.get(human_id)
        if pending is None or command_id not in pending:
            return
        pending.remove(command_id)
        state = self._states[human_id]
        state.metadata["anim_people_command_result"] = {
            "command_id": command_id,
            "command_name": str(payload["command_name"]),
            "status": str(payload["status"]),
            "time_code": float(payload["time_code"]),
        }
        # IRA 0.7.x reports natural completion as TaskStatus.default; explicit
        # failures and interruptions are different terminal outcomes.
        if payload["status"] != "default":
            state.action_status = "fail"
            state.last_error = (
                f"anim_people {payload['command_name']} ended with "
                f"status {payload['status']!r}."
            )

    @staticmethod
    def _last_agent_command_name(lines, human_id: str) -> str:
        """Return the command verb of the last agent-prefixed line in ``lines``.

        Non-agent-prefixed lines (e.g. ``Queue`` / ``Queue_Spot`` definitions)
        are skipped so the queue-contains check targets the character's actual
        enter action.
        """
        for line in reversed(lines):
            parts = str(line).strip().split()
            if len(parts) >= 2 and parts[0] == human_id:
                return parts[1]
        return ""

    def _fetch_character_script_instance(self, human_id: str) -> Any:
        try:
            from isaacsim.replicator.agent.core.agent_manager import AgentManager
            from omni.anim.people.scripts.utils import Utils

            instance = Utils.fetch_target_character_instance_by_name(human_id)
            expected_path = self._character_skelroot_paths.get(human_id)
            if instance is not None and (
                not expected_path
                or str(getattr(instance, "prim_path", "")) == expected_path
            ):
                return instance
            agent_manager = AgentManager.get_instance()
            if human_id not in getattr(agent_manager, "_agent_name_to_script_inst", {}):
                return None
            instance = agent_manager.get_agent_script_instance_by_name(human_id)
            if (
                expected_path
                and str(getattr(instance, "prim_path", "")) != expected_path
            ):
                return None
            return instance
        except Exception:
            return None

    @staticmethod
    def _command_name_from_line(command: str) -> str:
        parts = str(command).strip().split()
        return parts[1] if len(parts) > 1 else ""

    @staticmethod
    def _script_queue_contains_command(instance: Any, expected_command: str) -> bool:
        if not expected_command:
            return False
        for item in list(getattr(instance, "commands", []) or []):
            command = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if isinstance(command, str):
                parts = command.split()
            else:
                parts = [str(part) for part in command]
            if parts and parts[0] == expected_command:
                return True
        return False

    def _poll_character_position(self, human_id: str, state: HumanBodyState):
        """Read the native animated pose; authored USD is not live motion."""
        try:
            runtime_pose = self._poll_character_runtime_pose(human_id)
            if runtime_pose is None:
                raise RuntimeError("Native AnimPeople pose is unavailable.")
            position, yaw_rad, metadata = runtime_pose
            if yaw_rad is None:
                raise RuntimeError("Native AnimPeople orientation is unavailable.")
            state.position = position
            state.yaw_rad = yaw_rad
            state.metadata.update(metadata)
            state.pose_stale = False
            state.metadata["pose_stale"] = False
            state.metadata.pop("pose_poll_error", None)
        except Exception as exc:
            state.pose_stale = True
            state.metadata["pose_stale"] = True
            state.metadata["pose_poll_error"] = str(exc)
            logger.warning(
                f"[AnimPeopleBodyExecutor] Cannot poll position for "
                f"{human_id}: {exc}. Keeping last known state."
            )

        # Check if the character has reached its target
        if (
            state.target_position is not None
            and state.action_status == "ongoing"
            and not state.pose_stale
        ):
            current = np.asarray(state.position[:2], dtype=float)
            target = np.asarray(state.target_position[:2], dtype=float)
            distance = float(np.linalg.norm(target - current))
            # A near-target pose while IRA is still decelerating/turning does
            # not finish the leg. Wait for its native end event as well, so a
            # following Sit cannot be scheduled before walking actually ends.
            if (
                distance <= HUMAN_GOTO_ARRIVAL_TOLERANCE_M
                and not self._pending_command_ids.get(human_id)
            ):
                self._last_injected_goto.pop(human_id, None)
                pending = self._goto_waypoint_queues.get(human_id) or []
                if pending:
                    target = pending.pop(0)
                    self._goto_waypoint_queues[human_id] = pending
                    self._inject_command(
                        human_id,
                        f"{human_id} GoTo {target[0]} {target[1]} {target[2]} _",
                    )
                    self._last_injected_goto[human_id] = (
                        target[0],
                        target[1],
                    )
                    state.target_position = list(target)
                    state.animation_state = "goto_pending"
                else:
                    self._goto_waypoint_queues.pop(human_id, None)
                    state.animation_state = "idle"
                    state.action_status = "succeed"
                    state.target_position = None
            elif not self._pending_command_ids.get(human_id):
                state.action_status = "fail"
                state.last_error = (
                    f"GoTo ended {distance:.3f} m from its current route target, "
                    f"outside {HUMAN_GOTO_ARRIVAL_TOLERANCE_M:.3f} m arrival tolerance."
                )

    def _poll_character_runtime_pose(
        self,
        human_id: str,
    ) -> Optional[tuple[list[float], Optional[float], Dict[str, Any]]]:
        """Read official AnimPeople runtime pose from the BehaviorScript.

        GoTo locomotion is driven by the anim graph character object.  In Isaac
        People 0.7.x that runtime transform is not guaranteed to be mirrored to
        a USD xform prim every frame, so the BehaviorScript/AgentManager pose is
        the authoritative source for ego-camera following.
        """
        try:
            from isaacsim.replicator.agent.core.agent_manager import AgentManager
            from omni.anim.people.scripts.utils import Utils
        except Exception:
            return None

        instance = self._fetch_character_script_instance(human_id)
        if instance is None:
            return None

        position = None
        position_method = ""
        get_current_position = getattr(instance, "get_current_position", None)
        if callable(get_current_position):
            try:
                position = self._vec3_to_float_list(get_current_position())
                position_method = "behavior_script_get_current_position"
            except Exception:
                position = None

        character = getattr(instance, "character", None)
        if position is None and character is not None:
            try:
                position = self._vec3_to_float_list(Utils.get_character_pos(character))
                position_method = "utils_get_character_pos"
            except Exception:
                position = None
        if position is None:
            return None

        yaw_rad = None
        rotation_method = ""
        if character is not None:
            try:
                yaw_rad = self._runtime_quat_to_yaw_rad(
                    Utils.get_character_rot(character),
                    Utils,
                )
                rotation_method = "utils_get_character_rot"
            except Exception:
                yaw_rad = None

        script_prim_path = str(getattr(instance, "prim_path", "") or "")
        metadata: Dict[str, Any] = {
            "pose_source": "anim_people_runtime",
            "pose_prim_path": script_prim_path,
            "anim_people_script_prim_path": script_prim_path,
            "pose_position_method": position_method,
            "anim_people_current_command": None,
        }
        if rotation_method:
            metadata["pose_rotation_method"] = rotation_method
        current_command = getattr(instance, "current_command", None)
        get_command_name = getattr(current_command, "get_command_name", None)
        if callable(get_command_name):
            try:
                metadata["anim_people_current_command"] = get_command_name()
            except Exception:
                pass
        pending_commands = self._command_queue_snapshot(
            getattr(instance, "commands", [])
        )
        metadata["anim_people_pending_commands"] = pending_commands
        return position, yaw_rad, metadata

    @staticmethod
    def _vec3_to_float_list(value: Any) -> list[float]:
        if value is None:
            raise ValueError("Pose vector is None.")
        if all(hasattr(value, axis) for axis in ("x", "y", "z")):
            result = [float(value.x), float(value.y), float(value.z)]
        else:
            items = list(value)
            if len(items) < 3:
                raise ValueError(f"Pose vector has fewer than 3 values: {value!r}")
            result = [float(items[0]), float(items[1]), float(items[2])]
        if not np.all(np.isfinite(np.asarray(result, dtype=float))):
            raise ValueError(f"Pose vector has non-finite values: {result!r}")
        return result

    @staticmethod
    def _runtime_quat_to_yaw_rad(rotation: Any, utils_module: Any = None) -> float:
        if utils_module is not None and callable(
            getattr(utils_module, "convert_to_angle", None)
        ):
            angle_deg = float(utils_module.convert_to_angle(rotation))
            if np.isfinite(angle_deg):
                return float(math.radians(angle_deg))

        if all(hasattr(rotation, axis) for axis in ("x", "y", "z", "w")):
            x = float(rotation.x)
            y = float(rotation.y)
            z = float(rotation.z)
            w = float(rotation.w)
        else:
            values = list(rotation)
            if len(values) < 4:
                raise ValueError(
                    f"Runtime quaternion has fewer than 4 values: {rotation!r}"
                )
            x, y, z, w = (
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
            )
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        if not math.isfinite(yaw):
            raise ValueError(
                f"Runtime quaternion produced non-finite yaw: {rotation!r}"
            )
        return float(yaw)

    @staticmethod
    def _command_queue_snapshot(commands: Any, *, limit: int = 3) -> list[Any]:
        snapshot = []
        for item in list(commands or [])[:limit]:
            command = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if isinstance(command, (list, tuple)):
                snapshot.append([str(part) for part in command])
            else:
                snapshot.append(str(command))
        return snapshot

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_duration(action: Dict[str, Any], default: float) -> float:
        duration = float(action.get("duration", default))
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(
                "Human animation duration must be finite and positive seconds."
            )
        return duration

    def _enable_sit_pose(self, human_id: str, state: HumanBodyState) -> None:
        """Select the full Sit clip in Biped_Setup's Sit-state blend.

        The shipped graph connects SitWeight to Blend(pose0=filtered idle,
        pose1=Sit), but authors no default. The native Sit command sets Action
        only; accepting it does not establish a nonzero clip weight.
        """
        instance = self._fetch_character_script_instance(human_id)
        character = getattr(instance, "character", None)
        if character is None:
            raise RuntimeError(f"Human {human_id!r} has no live animation character.")
        previous = character.get_variable("SitWeight")
        character.set_variable("SitWeight", 1.0)
        actual = character.get_variable("SitWeight")
        # AnimGraph's native getter returns a list even for a float variable.
        # A missing or multi-element value is not a scalar Sit blend weight.
        if not isinstance(actual, list) or len(actual) != 1:
            raise RuntimeError(
                f"Human {human_id!r} SitWeight must return one scalar value: {actual!r}"
            )
        if not math.isclose(float(actual[0]), 1.0, abs_tol=1e-6):
            raise RuntimeError(
                f"Human {human_id!r} SitWeight did not become 1.0: {actual!r}"
            )
        state.metadata["sit_blend_weight_before"] = previous
        state.metadata["sit_blend_weight"] = float(actual[0])

    @staticmethod
    def _ensure_seat_interact_offsets(seat_prim_path: str) -> Dict[str, Any]:
        """Author the child xforms ``Sit`` needs, if the seat asset lacks them.

        omni.anim.people's ``Sit`` does not sit on the seat prim.  It walks to
        ``<seat>/walk_to_offset`` and only plays the sit animation once that
        walk converges; ``interact_offset`` is where the body is then lerped
        to.  When those children are absent
        ``InteractableObjectHelper.get_interact_prim_offsets`` silently falls
        back to the seat prim's *own* transform.

        For GRScenes furniture that origin lies inside the mesh, which is not a
        standable NavMesh point, so ``destination_reached()`` never fires: the
        character orbits the seat forever, ``walk()`` never returns True, and
        the sit animation never plays.  Measured on the MVUCSQ... stool, the
        Human held 0.76-0.88 m from the origin for the whole 25 s window while
        reporting ``animation_state='sitting'`` -- the state field describes the
        command that was injected, not the body, so this failure is invisible
        without looking at the video or the positions.

        The walk target is snapped onto the NavMesh, which is what makes it
        reachable.

        ``interact_offset`` is deliberately kept at FLOOR level, in front of the
        seat, facing away from it.  The Sit clip does not translate the
        character root: measured on ``Sit_skelanim`` in ``Biped_Setup.usd``, the
        pelvis is driven from 0.957 down to 0.610 and 0.397 backwards over the
        clip, entirely through joint rotations, while the root stays put.  The
        clip therefore already assumes the root is standing on the floor with
        the seat *behind* the character, and lowers the body onto it by itself.
        (Toes point +Z in the biped's Y-up frame and the pelvis travels -Z, so
        the motion is backwards, not forwards.)

        Placing ``interact_offset`` on the seat top instead -- the obvious
        reading of "where the character sits" -- adds a seat height underneath
        a pose that has already sat down, and the character ends up standing on
        the furniture.  That was measured, not theorised: the Human was raised
        to z=0.053 for ~4.3 s, matching the 5 s sit duration, with straight legs
        throughout.

        Lengths are handled in metres and converted through the stage's
        ``metersPerUnit``.  GRScenes source layers are authored in centimetres,
        so a bare 0.45 would be 4.5 mm there -- an approach point inside the
        seat, i.e. exactly the bug this function exists to remove.
        """
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        report: Dict[str, Any] = {"seat_prim": seat_prim_path}
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable.")
        seat = stage.GetPrimAtPath(seat_prim_path)
        if seat is None or not seat.IsValid():
            raise RuntimeError(f"Seat prim is missing: {seat_prim_path}")

        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
        if meters_per_unit <= 0.0:
            raise RuntimeError(
                f"Stage reports a non-positive metersPerUnit "
                f"({meters_per_unit}); seat offsets cannot be placed."
            )
        units_per_meter = 1.0 / meters_per_unit
        report["meters_per_unit"] = meters_per_unit

        had_walk = stage.GetPrimAtPath(f"{seat_prim_path}/walk_to_offset").IsValid()
        had_interact = stage.GetPrimAtPath(
            f"{seat_prim_path}/interact_offset"
        ).IsValid()
        report["walk_to_offset_preexisting"] = bool(had_walk)
        report["interact_offset_preexisting"] = bool(had_interact)
        if had_walk != had_interact:
            raise RuntimeError(
                f"Seat {seat_prim_path} must author both walk_to_offset and "
                "interact_offset; a partial binding cannot preserve seating alignment."
            )
        if had_walk and had_interact:
            report["authored"] = False
            for name in ("walk_to_offset", "interact_offset"):
                matrix = omni.usd.get_world_transform_matrix(
                    stage.GetPrimAtPath(f"{seat_prim_path}/{name}")
                )
                report[f"{name}_world"] = [
                    float(value) for value in matrix.ExtractTranslation()
                ]
                report[f"{name}_world_matrix"] = [list(row) for row in matrix]
            separation_m = meters_per_unit * math.dist(
                report["walk_to_offset_world"], report["interact_offset_world"]
            )
            if separation_m > SEAT_STANCE_TOLERANCE_M:
                raise RuntimeError(
                    f"Seat offsets differ by {separation_m:.4f} m; native Sit would slide its root. "
                    "Place both offsets at the verified standing anchor before admission."
                )
            walk_matrix = report["walk_to_offset_world_matrix"]
            interact_matrix = report["interact_offset_world_matrix"]
            if (
                yaw_error_rad(
                    math.atan2(walk_matrix[0][1], walk_matrix[0][0]),
                    math.atan2(interact_matrix[0][1], interact_matrix[0][0]),
                )
                > SEAT_YAW_TOLERANCE_RAD
            ):
                raise RuntimeError("Seat offsets disagree on the standing orientation.")
            return report

        # Seat extent drives both the approach distance and the sit height.
        bbox = (
            UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            )
            .ComputeWorldBound(seat)
            .ComputeAlignedRange()
        )
        if bbox.IsEmpty():
            raise RuntimeError(
                f"Seat prim has an empty world bound, so no walk_to_offset can "
                f"be derived: {seat_prim_path}"
            )
        centre = bbox.GetMidpoint()
        seat_top_z = float(bbox.GetMax()[2])
        radius_m = (
            meters_per_unit
            * 0.5
            * max(
                float(bbox.GetMax()[0] - bbox.GetMin()[0]),
                float(bbox.GetMax()[1] - bbox.GetMin()[1]),
            )
        )

        # Stand off far enough to clear the seat, then snap onto the NavMesh so
        # the walk can actually converge.  Try several bearings: the first one
        # may land inside a wall or another piece of furniture.
        approach_m = max(radius_m + 0.45, 0.60)
        approach_units = approach_m * units_per_meter
        walk_xy: Optional[List[float]] = None
        floor_z: Optional[float] = None
        snapped_from: Optional[List[float]] = None
        for index in range(8):
            angle = 2.0 * float(np.pi) * index / 8.0
            candidate = [
                float(centre[0]) + approach_units * float(np.cos(angle)),
                float(centre[1]) + approach_units * float(np.sin(angle)),
                0.0,
            ]
            try:
                from omni.anim.people.scripts.utils import Utils

                snapped = Utils.get_closest_navmesh_point(candidate)
            except Exception:
                snapped = None
            if snapped is None:
                continue
            snapped_xy = [float(snapped[0]), float(snapped[1])]
            # A snap that travels far means the bearing was blocked and the
            # NavMesh threw the point somewhere unrelated; reject it.
            drift_m = (
                meters_per_unit
                * (
                    (snapped_xy[0] - candidate[0]) ** 2
                    + (snapped_xy[1] - candidate[1]) ** 2
                )
                ** 0.5
            )
            if drift_m <= 0.35:
                walk_xy = snapped_xy
                floor_z = float(snapped[2])
                snapped_from = candidate[:2]
                report["approach_bearing_index"] = index
                report["navmesh_snap_drift_m"] = round(float(drift_m), 4)
                break
        if walk_xy is None or floor_z is None:
            raise RuntimeError(
                f"No reachable approach point found around {seat_prim_path}; "
                f"the seat is enclosed or off the NavMesh, so Sit would orbit "
                f"it forever instead of failing."
            )
        report["walk_to_offset_world"] = [walk_xy[0], walk_xy[1], floor_z]
        report["candidate_before_snap"] = snapped_from

        # The character must end up with its back to the seat, because the clip
        # moves the pelvis *backwards*.  ``away`` is the outward bearing from
        # the seat centre to the approach point, which is exactly that facing.
        away = [walk_xy[0] - float(centre[0]), walk_xy[1] - float(centre[1])]
        away_length = (away[0] ** 2 + away[1] ** 2) ** 0.5
        if away_length <= 0.0:
            raise RuntimeError(
                f"Approach point coincides with the seat centre for "
                f"{seat_prim_path}; no sit facing can be derived."
            )
        away = [away[0] / away_length, away[1] / away_length]
        face_deg = math.degrees(math.atan2(away[1], away[0]))
        # ``walk_to_rot`` is compared against the character's own world rotation
        # through ``Utils.convert_to_angle`` on both sides, so what is authored
        # here is a character-style yaw, not a bare heading.  Measured on a
        # steady walk segment: course - yaw = -90 deg.
        yaw_deg = face_deg + AnimPeopleBodyExecutor.CHARACTER_FACING_YAW_OFFSET_DEG
        back_travel_units = (
            AnimPeopleBodyExecutor.PELVIS_SIT_BACK_TRAVEL_M * units_per_meter
        )

        # Children are authored in the seat's local frame, so undo its
        # world transform before writing the transforms.
        seat_world = omni.usd.get_world_transform_matrix(seat)
        try:
            to_local = seat_world.GetInverse()
        except Exception as exc:
            raise RuntimeError(
                f"Seat transform is not invertible: {seat_prim_path}"
            ) from exc

        def _author(name: str, world_xy: List[float], world_yaw_deg: float) -> None:
            world = Gf.Matrix4d(1.0)
            world.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(world_yaw_deg)))
            world.SetTranslateOnly(Gf.Vec3d(world_xy[0], world_xy[1], floor_z))
            child = UsdGeom.Xform.Define(stage, f"{seat_prim_path}/{name}")
            child.ClearXformOpOrder()
            # One transform op rather than translate+rotate: it reproduces the
            # world pose exactly even when the seat carries inherited scale.
            child.AddTransformOp().Set(world * to_local)

        if not had_interact:
            # Floor level, one pelvis-back-travel out from the seat centre along
            # the approach bearing.  The clip drops and reverses the pelvis by
            # itself with no root translation, so the root belongs on the floor;
            # authoring the seat top here stands the character on the furniture.
            interact_xy = [
                float(centre[0]) + away[0] * back_travel_units,
                float(centre[1]) + away[1] * back_travel_units,
            ]
            from omni.anim.people.scripts.utils import Utils

            stance = [interact_xy[0], interact_xy[1], floor_z]
            snapped_stance = Utils.get_closest_navmesh_point(stance)
            if (
                snapped_stance is None
                or meters_per_unit
                * math.dist(stance, [float(snapped_stance[i]) for i in range(3)])
                > SEAT_STANCE_TOLERANCE_M
            ):
                raise RuntimeError(
                    "Seat standing anchor is not reachable on the native NavMesh."
                )
            # Walk must reach the stance that the clip actually uses. A remote
            # approach followed by a root lerp is not a seating transfer.
            _author("walk_to_offset", interact_xy, yaw_deg)
            report["walk_to_offset_world"] = list(stance)
            _author("interact_offset", interact_xy, yaw_deg)
            report["interact_offset_world"] = [
                round(interact_xy[0], 4),
                round(interact_xy[1], 4),
                floor_z,
            ]

        report["authored"] = True
        report["seat_radius_m"] = round(float(radius_m), 4)
        report["approach_distance_m"] = round(float(approach_m), 4)
        report["seat_top_z"] = round(float(seat_top_z), 4)
        report["facing_deg"] = round(float(face_deg), 2)
        report["authored_yaw_deg"] = round(float(yaw_deg), 2)
        report["pelvis_sit_back_travel_m"] = (
            AnimPeopleBodyExecutor.PELVIS_SIT_BACK_TRAVEL_M
        )
        return report

    def _sync(self, human_id: str):
        state = self._states[human_id]
        observed = (
            state.animation_state,
            state.active_action,
            state.action_status,
            self._physics_step is not None,
        )
        if self._last_recorded_state.get(human_id) != observed:
            self._replay_log.append(
                {
                    "event": "human_body_state_changed",
                    "physics_step": self._physics_step,
                    "human_id": human_id,
                    "animation_state": state.animation_state,
                    "active_action": state.active_action,
                    "action_status": state.action_status,
                }
            )
            self._last_recorded_state[human_id] = observed
        if self._pose_sync is not None:
            self._pose_sync(human_id, state)

    def _resolve_character_name(self, character_name: Optional[str]) -> str:
        return str(character_name or "").strip() or self._default_character

    def _resolve_character_usd(self, character_name: str) -> Path:
        if not self._character_assets_path:
            raise FileNotFoundError(
                "No Isaac People character asset path configured for AnimPeopleBodyExecutor."
            )
        from tarsphere.humans.character_catalog import resolve_character_usd

        return resolve_character_usd(
            character_name,
            character_assets_root=self._character_assets_path,
        )

    @staticmethod
    def _pump_app_updates(count: int = 3):
        from tarsphere.infra.simulation.kit_updates import pump_kit_updates

        pump_kit_updates(count=count)

    @staticmethod
    def _valid_target(target: Any) -> bool:
        return target is not None and hasattr(target, "__len__") and len(target) >= 2

    def _goto_is_redundant(
        self, state: HumanBodyState, target_position: list[float]
    ) -> bool:
        """True if a GoTo to ``target_position`` should NOT be (re-)injected.

        Two cases:
        (a) the character is already at the target (within reach) — mark it
            arrived and skip injection;
        (b) the character is already walking to essentially the same target —
            let the in-flight GoTo continue.

        Prevents the degenerate-path / queue-growth re-injection described in
        the ``goto`` branch.
        """
        target_xy = np.asarray(target_position[:2], dtype=float)
        if not getattr(state, "pose_stale", False):
            current_xy = np.asarray(state.position[:2], dtype=float)
            if (
                float(np.linalg.norm(target_xy - current_xy))
                <= HUMAN_GOTO_ARRIVAL_TOLERANCE_M
            ):
                if state.active_action == "goto":
                    state.action_status = "succeed"
                    state.animation_state = "idle"
                    state.target_position = None
                return True
        if (
            state.active_action == "goto"
            and state.action_status == "ongoing"
            and state.target_position is not None
        ):
            active_xy = np.asarray(state.target_position[:2], dtype=float)
            if (
                float(np.linalg.norm(target_xy - active_xy))
                < self.GOTO_COMMAND_DEDUP_TOLERANCE_M
            ):
                return True
        return False

    def _goto_already_injected(
        self, human_id: str, target_position: list[float]
    ) -> bool:
        """True if an equivalent GoTo was already injected and not yet arrived.

        Pose read-back (used by ``_goto_is_redundant``) can be stale mid-walk,
        so this guards purely on the last injected destination — robust against
        a stateless policy that re-decides the same GoTo every step.  Cleared
        on arrival (``_poll_character_position``) and on reset, so a genuine
        later return to the same destination is still injected.
        """
        state = self._states[human_id]
        if state.active_action != "goto" or state.action_status != "ongoing":
            return False
        last = self._last_injected_goto.get(human_id)
        if last is None:
            return False
        delta = np.asarray(target_position[:2], dtype=float) - np.asarray(
            last, dtype=float
        )
        return float(np.linalg.norm(delta)) < self.GOTO_COMMAND_DEDUP_TOLERANCE_M

    def _failed(self, human_id: str, command: str, feedback: str) -> HumanBodyResult:
        state = self._states.get(human_id)
        if state is not None:
            state.action_status = "fail"
            state.last_error = feedback
        return HumanBodyResult(human_id, command, False, feedback=feedback, state=state)
