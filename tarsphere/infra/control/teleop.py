"""Teleoperation commands, keyboard state, and actor control frames.

This module is Isaac-free so the command contract can be unit-tested without
a GPU session. The runner owns Kit, ``env.step``, and keyboard sampling.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tarsphere.infra.control.physical_contracts import (
    CAMERA_GAZE_TILT_LIMIT_DEG,
    G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
    HUMAN_CHARACTER_YAW_OFFSET_RAD,
)

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
DEFAULT_CROUCH_PELVIS_HEIGHT_M = 0.55
HEADLESS_EGO_NOTICE = (
    "Headless mode has no viewport; robot-eye sensors remain available."
)

HELP = """Commands:
  robot
  human [id]
  ego on | ego off
  crouch [pelvis_height_m]
  stand
  tilt <pitch_deg>
  move <forward_m> <left_m> <yaw_deg>
  wait <seconds>
  grasp <phrase>
  capture [label]
  goto <human_id> <x> <y> [z] [end_rot]
  idle <human_id>
  look <human_id> [duration_s]
  sit <human_id> <seat_prim_path> [duration_s]
  list
  status
  help
  quit

Isaac window (focus the viewport): I/K forward/back, J/L strafe, U/O yaw.
F8 or capture [label]: save one still frame; labels use letters, digits, _ or -.
Keyboard locomotion drives only the active agent. Human U/O changes the next walking direction.
Headless: no I/K and no viewport switch. stdin skills still work; grasp uses observe().
grasp <phrase> directly runs visual grounding and physical grasp execution.
"""


@dataclass(frozen=True)
class TeleopCommand:
    """One parsed stdin line for the teleoperation runner.

    ``kind`` is the dispatch key the runner switches on. ``payload`` holds
    only the arguments that kind consumes; unused keys stay absent.
    """

    kind: str
    raw: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IdleKeyboardCommand:
    """Zero locomotion vector when Kit has no keyboard subscription."""

    def __getitem__(self, index: int) -> float:
        if index < 0 or index > 5:
            raise IndexError(index)
        return 0.0

    def tolist(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TeleopKeyboardState:
    """Combine held motion keys and retain short U/O taps in simulation time.

    Kit may deliver a press and its release during one render update. The
    minimum turn interval starts when ``sample`` first consumes that press,
    so slow rendering or a paused timeline cannot discard it beforehand.
    """

    _KEYS = ("I", "K", "J", "L", "U", "O")

    def __init__(self, minimum_turn_seconds: float = 0.15) -> None:
        self.minimum_turn_seconds = float(minimum_turn_seconds)
        if (
            not math.isfinite(self.minimum_turn_seconds)
            or self.minimum_turn_seconds < 0.15
        ):
            raise ValueError("minimum_turn_seconds must be finite and at least 0.15.")
        self.clear()

    def clear(self) -> None:
        """Discard held keys and unconsumed turns when control changes."""

        self.held: set[str] = set()
        self._pending_turn = 0
        self._pulse_direction = 0
        self._pulse_until = 0.0

    def _key(self, key: str) -> str:
        resolved = str(key).upper()
        if resolved not in self._KEYS:
            raise ValueError(f"Unknown motion key: {key!r}")
        return resolved

    def press(self, key: str) -> None:
        """Record one key; OS repeat does not restart its turn pulse."""

        key = self._key(key)
        if key not in self.held and key in {"U", "O"}:
            self._pending_turn = 1 if key == "U" else -1
        self.held.add(key)

    def release(self, key: str) -> None:
        """Release only this key, preserving other held motion axes."""

        self.held.discard(self._key(key))

    def sample(self, simulation_time: float) -> list[float]:
        """Return the six motion channels in I/K/J/L/U/O order."""

        simulation_time = float(simulation_time)
        if not math.isfinite(simulation_time):
            raise ValueError("simulation_time must be finite.")
        if self._pending_turn:
            self._pulse_direction = self._pending_turn
            self._pulse_until = simulation_time + self.minimum_turn_seconds
            self._pending_turn = 0
        if {"U", "O"} <= self.held:
            self._pulse_until = simulation_time
            yaw = 0
        elif "U" in self.held:
            yaw = 1
        elif "O" in self.held:
            yaw = -1
        else:
            yaw = self._pulse_direction if simulation_time < self._pulse_until else 0
        return [
            *(float(key in self.held) for key in self._KEYS[:4]),
            float(yaw > 0),
            float(yaw < 0),
        ]


@dataclass
class TeleopSession:
    """Mutable control state for one teleoperation run.

    ``active_agent`` is who I/K/J/L/U/O currently drive. ``active_human_id``
    is the human that ``human`` (with no id) and keyboard human loco use.
    ``viewer_mode`` tracks the intended viewport, not whether ego sensors exist.
    ``headless`` means there is no Kit window; stdin remains the control path.
    """

    active_agent: str = "robot"
    active_human_id: str = ""
    viewer_mode: str = "third_person"
    headless: bool = False
    grasp_enabled: bool = False

    def switch_to_robot(self) -> None:
        self.active_agent = "robot"

    def switch_to_human(
        self,
        human_id: str | None,
        *,
        known_human_ids: Sequence[str],
        default_human_id: str,
    ) -> str:
        resolved = str(human_id or default_human_id or "").strip()
        if not resolved:
            if not known_human_ids:
                raise ValueError("Episode has no humans to control.")
            resolved = str(known_human_ids[0])
        if resolved not in known_human_ids:
            raise ValueError(
                f"Unknown human_id {resolved!r}; valid={list(known_human_ids)}"
            )
        self.active_agent = "human"
        self.active_human_id = resolved
        return resolved

    def set_viewer_mode(self, mode: str) -> str:
        if mode not in {"first_person", "third_person"}:
            raise ValueError(
                f"viewer mode must be 'first_person' or 'third_person'; got {mode!r}."
            )
        self.viewer_mode = mode
        return mode


def resolve_ego_viewport(
    session: TeleopSession,
    viewer_mode: str,
) -> tuple[str, bool, str | None]:
    """Record the requested viewer mode and say whether Kit should switch.

    Headless Isaac has no viewport; ``IsaacAutonomousEnv.set_viewer_mode``
    raises. Ego RGB-D sensors still exist when the env was constructed with
    ``enable_ego_sensors=True``, so grasp continues to use ``observe()``.
    """

    mode = session.set_viewer_mode(viewer_mode)
    if session.headless:
        return mode, False, HEADLESS_EGO_NOTICE
    return mode, True, None


def parse_command(
    line: str,
    *,
    known_human_ids: Sequence[str] | None = None,
    default_human_id: str = "",
) -> TeleopCommand:
    """Parse one stdin line into a ``TeleopCommand``.

    Unknown verbs and malformed arguments raise ``ValueError``. Empty input is
    a no-op. Human existence is checked only when ``known_human_ids`` is given.
    """

    raw = line.strip()
    if not raw:
        return TeleopCommand(kind="noop", raw=raw)
    parts = raw.split()
    verb = parts[0].lower()

    if verb in {"quit", "exit", "q"}:
        return TeleopCommand(kind="quit", raw=raw)
    if verb in {"help", "?"}:
        return TeleopCommand(kind="help", raw=raw)
    if verb == "list":
        return TeleopCommand(kind="list", raw=raw)
    if verb == "status":
        return TeleopCommand(kind="status", raw=raw)
    if verb == "capture":
        if len(parts) > 2:
            raise ValueError("capture accepts one optional filename label.")
        label = parts[1] if len(parts) == 2 else "frame"
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", label) is None:
            raise ValueError(
                "capture label must be 1..64 letters, digits, underscores or hyphens."
            )
        return TeleopCommand(kind="capture", raw=raw, payload={"label": label})
    if verb == "robot":
        if len(parts) != 1:
            raise ValueError("robot takes no arguments.")
        return TeleopCommand(kind="switch_robot", raw=raw)
    if verb == "human":
        if len(parts) > 2:
            raise ValueError("human accepts at most one id.")
        human_id = parts[1] if len(parts) == 2 else None
        if human_id is not None:
            _require_known_human(human_id, known_human_ids)
        return TeleopCommand(
            kind="switch_human",
            raw=raw,
            payload={"human_id": human_id or default_human_id or None},
        )
    if verb == "ego":
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            raise ValueError("ego requires `on` or `off`.")
        enabled = parts[1].lower() == "on"
        return TeleopCommand(
            kind="ego",
            raw=raw,
            payload={
                "enabled": enabled,
                "viewer_mode": "first_person" if enabled else "third_person",
            },
        )
    if verb == "crouch":
        if len(parts) > 2:
            raise ValueError("crouch accepts at most one pelvis height in metres.")
        height = (
            _finite_number(parts[1], name="pelvis_height_m")
            if len(parts) == 2
            else DEFAULT_CROUCH_PELVIS_HEIGHT_M
        )
        if not (
            G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M
            <= height
            <= G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M
        ):
            raise ValueError(
                "crouch pelvis_height_m must be between "
                f"{G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M} and "
                f"{G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M} metres; got {height}."
            )
        return TeleopCommand(
            kind="crouch",
            raw=raw,
            payload={"pelvis_height_m": height},
        )
    if verb == "stand":
        if len(parts) != 1:
            raise ValueError("stand takes no arguments.")
        return TeleopCommand(kind="stand", raw=raw)
    if verb == "tilt":
        if len(parts) != 2:
            raise ValueError("tilt requires one pitch in degrees.")
        pitch_deg = _finite_number(parts[1], name="pitch_deg")
        if not (-CAMERA_GAZE_TILT_LIMIT_DEG <= pitch_deg <= CAMERA_GAZE_TILT_LIMIT_DEG):
            raise ValueError(
                "tilt pitch_deg must be between "
                f"{-CAMERA_GAZE_TILT_LIMIT_DEG} and {CAMERA_GAZE_TILT_LIMIT_DEG}; "
                f"got {pitch_deg}."
            )
        return TeleopCommand(
            kind="tilt",
            raw=raw,
            payload={"pitch_deg": pitch_deg},
        )
    if verb == "grasp":
        phrase = raw[len(parts[0]) :].strip()
        if not phrase:
            raise ValueError(
                "grasp requires a visual phrase, e.g. `grasp white bottle`."
            )
        return TeleopCommand(
            kind="grasp",
            raw=raw,
            payload={"target": phrase},
        )
    if verb == "move":
        if len(parts) != 4:
            raise ValueError("move requires: move <forward_m> <left_m> <yaw_deg>")
        delta = [
            _finite_number(value, name=name)
            for value, name in zip(parts[1:], ("forward_m", "left_m", "yaw_deg"))
        ]
        return TeleopCommand(kind="move", raw=raw, payload={"delta": delta})
    if verb == "wait":
        if len(parts) != 2:
            raise ValueError("wait requires one duration in seconds.")
        seconds = _finite_number(parts[1], name="seconds")
        if seconds <= 0:
            raise ValueError("wait seconds must be positive.")
        return TeleopCommand(kind="wait", raw=raw, payload={"seconds": seconds})

    if verb == "goto":
        return _parse_human_goto(parts, raw=raw, known_human_ids=known_human_ids)
    if verb == "idle":
        if len(parts) != 2:
            raise ValueError("idle requires: idle <human_id>")
        human_id = _require_known_human(parts[1], known_human_ids)
        return TeleopCommand(
            kind="human_action",
            raw=raw,
            payload={"human_id": human_id, "action": {"command": "idle"}},
        )
    if verb in {"look", "look_around"}:
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError("look requires: look <human_id> [duration_s]")
        human_id = _require_known_human(parts[1], known_human_ids)
        duration = (
            _finite_number(parts[2], name="duration_s") if len(parts) == 3 else 5.0
        )
        if duration <= 0:
            raise ValueError("look duration_s must be positive.")
        return TeleopCommand(
            kind="human_action",
            raw=raw,
            payload={
                "human_id": human_id,
                "action": {"command": "look_around", "duration": duration},
            },
        )
    if verb == "sit":
        if len(parts) < 3 or len(parts) > 4:
            raise ValueError(
                "sit requires: sit <human_id> <seat_prim_path> [duration_s]"
            )
        human_id = _require_known_human(parts[1], known_human_ids)
        duration = (
            _finite_number(parts[3], name="duration_s") if len(parts) == 4 else 5.0
        )
        if duration <= 0:
            raise ValueError("sit duration_s must be positive.")
        return TeleopCommand(
            kind="human_action",
            raw=raw,
            payload={
                "human_id": human_id,
                "action": {
                    "command": "sit",
                    "seat_prim": parts[2],
                    "duration": duration,
                },
            },
        )
    raise ValueError(f"Unsupported command {verb!r}. Type help.")


def robot_tool_request(
    command: TeleopCommand,
) -> tuple[str, dict[str, Any]] | None:
    """Map a parsed command onto one registry tool call, or None."""

    if command.kind == "crouch":
        return "crouch", {"pelvis_height_m": float(command.payload["pelvis_height_m"])}
    if command.kind == "stand":
        return "stand_up", {}
    if command.kind == "tilt":
        return "tilt_camera", {"pitch_deg": float(command.payload["pitch_deg"])}
    if command.kind == "move":
        return "move_body", {"delta": list(command.payload["delta"])}
    if command.kind == "wait":
        return "wait", {"seconds": float(command.payload["seconds"])}
    if command.kind == "grasp":
        return "grasp_object", {"target": str(command.payload["target"])}
    return None


def require_grasp_stack(
    episode: Any,
    tools: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail loud when physical_task, SAM3, or GraspGenX is missing."""

    world = getattr(episode, "world_knowledge", None) or {}
    if not isinstance(world, dict):
        raise RuntimeError(
            "Teleoperation requires episode.world_knowledge to be a mapping "
            "that contains physical_task."
        )
    physical_task = world.get("physical_task")
    if not isinstance(physical_task, dict) or not physical_task:
        raise RuntimeError(
            "Teleoperation requires world_knowledge.physical_task so "
            "grasp_object has an authored target. Use "
            "episodes/case/fetch_and_deliver/base.yaml; "
            "episodes/case/navigate_and_report/human_coordination.yaml has no physical_task."
        )
    handler = getattr(tools, "handler", None)
    if not callable(handler) or handler("grasp_object") is None:
        raise RuntimeError(
            "grasp_object is not registered. build_robot_tool_registry must "
            "wire both SAM3 and GraspGenX; missing backends are not degraded."
        )
    return physical_task


def speed_from_keyboard(
    command: Any,
    *,
    forward_scale: float,
    strafe_scale: float,
    yaw_scale: float,
) -> list[float]:
    return [
        float(command[0] - command[1]) * float(forward_scale),
        float(command[2] - command[3]) * float(strafe_scale),
        float(command[4] - command[5]) * float(yaw_scale),
    ]


def keyboard_vector(command: Any) -> tuple[float, float, float]:
    return (
        float(command[0] - command[1]),
        float(command[2] - command[3]),
        float(command[4] - command[5]),
    )


def keyboard_is_active(command: Any) -> bool:
    forward, lateral, yaw = keyboard_vector(command)
    return bool(forward or lateral or yaw)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def human_heading_from_body_yaw(body_yaw: float) -> float:
    """Convert AnimPeople root yaw into the avatar's visible forward yaw."""

    return wrap_angle(float(body_yaw) - HUMAN_CHARACTER_YAW_OFFSET_RAD)


def human_end_rotation_deg(heading_yaw: float) -> float:
    """Convert a visible heading into the native GoTo root-yaw degrees."""

    root_yaw = wrap_angle(float(heading_yaw) + HUMAN_CHARACTER_YAW_OFFSET_RAD)
    return round(math.degrees(root_yaw), 3)


@dataclass
class HumanKeyboardHeading:
    """One actor's persistent keyboard frame, independent of other actors."""

    yaw_rad: float
    body_offset_rad: float = 0.0

    def commit_goto(self) -> None:
        """Consume the turn offset once a native GoTo accepts its heading.

        The body-follow frame will read the native actor's resulting yaw on
        subsequent updates. Keeping the consumed offset would apply the same
        operator turn again. World-follow mode retains ``yaw_rad`` unchanged.
        Rejected commands must not call this method.
        """

        self.body_offset_rad = 0.0

    def update(self, *, body_yaw: float, yaw_delta: float, follow_body: bool) -> float:
        if follow_body:
            self.body_offset_rad = wrap_angle(self.body_offset_rad + yaw_delta)
            self.yaw_rad = wrap_angle(
                human_heading_from_body_yaw(body_yaw) + self.body_offset_rad
            )
        else:
            self.yaw_rad = wrap_angle(self.yaw_rad + yaw_delta)
        return self.yaw_rad


def human_movement_heading(
    heading_yaw: float,
    *,
    forward: float,
    lateral: float,
) -> float:
    """Return the world course selected by forward/backward/strafe input."""

    dx = math.cos(heading_yaw) * forward - math.sin(heading_yaw) * lateral
    dy = math.sin(heading_yaw) * forward + math.cos(heading_yaw) * lateral
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return wrap_angle(heading_yaw)
    return math.atan2(dy, dx)


def can_inject_human_step(
    state: Mapping[str, Any],
    *,
    allow_replan_while_walking: bool,
) -> bool:
    if allow_replan_while_walking:
        return True
    return not (
        state.get("active_action") in {"move", "goto"}
        and state.get("action_status") == "ongoing"
        and state.get("target_position") is not None
    )


def target_from_command(
    position: Sequence[float],
    heading_yaw: float,
    *,
    forward: float,
    lateral: float,
    stride_m: float,
) -> list[float]:
    forward_x = math.cos(heading_yaw)
    forward_y = math.sin(heading_yaw)
    left_x = -math.sin(heading_yaw)
    left_y = math.cos(heading_yaw)
    delta_x = (forward * forward_x + lateral * left_x) * float(stride_m)
    delta_y = (forward * forward_y + lateral * left_y) * float(stride_m)
    return [
        float(position[0] + delta_x),
        float(position[1] + delta_y),
        float(position[2] if len(position) >= 3 else 0.0),
    ]


def _parse_human_goto(
    parts: Sequence[str],
    *,
    raw: str,
    known_human_ids: Sequence[str] | None,
) -> TeleopCommand:
    if len(parts) < 4:
        raise ValueError("goto requires: goto <human_id> <x> <y> [z] [end_rot]")
    human_id = _require_known_human(parts[1], known_human_ids)
    x = _finite_number(parts[2], name="x")
    y = _finite_number(parts[3], name="y")
    z: float | None
    if len(parts) >= 5 and parts[4] != "_":
        z = _finite_number(parts[4], name="z")
    else:
        z = None
    end_rot: float | str
    if len(parts) >= 6 and parts[5] != "_":
        end_rot = _finite_number(parts[5], name="end_rot")
    else:
        end_rot = "_"
    if len(parts) > 6:
        raise ValueError("goto accepts at most: goto <human_id> <x> <y> [z] [end_rot]")
    return TeleopCommand(
        kind="human_action",
        raw=raw,
        payload={
            "human_id": human_id,
            "action": {
                "command": "goto",
                "target": [x, y, z],
                "end_rot": end_rot,
            },
        },
    )


def _require_known_human(
    human_id: str,
    known_human_ids: Sequence[str] | None,
) -> str:
    if known_human_ids is not None and human_id not in known_human_ids:
        raise ValueError(
            f"Unknown human_id {human_id!r}; valid={list(known_human_ids)}"
        )
    return human_id


def _finite_number(raw: str, *, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {raw!r}.")
    return value
