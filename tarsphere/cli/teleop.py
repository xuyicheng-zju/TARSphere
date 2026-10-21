"""Control a robot and scene residents through one Isaac Sim session."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import select
import sys
import time
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tarsphere.harness.agent import agent_observation_from_runtime
from tarsphere.harness.core.types import ToolCall
from tarsphere.infra.control.teleop import (
    HELP,
    TeleopSession,
    IdleKeyboardCommand,
    TeleopKeyboardState,
    HumanKeyboardHeading,
    human_heading_from_body_yaw,
    human_end_rotation_deg,
    human_movement_heading,
    can_inject_human_step,
    keyboard_vector,
    parse_command,
    require_grasp_stack,
    resolve_ego_viewport,
    robot_tool_request,
    speed_from_keyboard,
    target_from_command,
)
from tarsphere.infra.control.physical_contracts import SIMULATION_PHYSICS_HZ
from tarsphere.infra.action_errors import ActionRefusedError

if TYPE_CHECKING:
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv


def _human_ids(env: IsaacAutonomousEnv) -> list[str]:
    return list(env._human_positions.keys())


def _human_state(env: IsaacAutonomousEnv, human_id: str) -> dict[str, Any]:
    snapshot = env.get_world_snapshot([human_id])
    return dict((snapshot.get("humans") or {}).get(human_id) or {})


def _human_observations(env: IsaacAutonomousEnv) -> dict[str, Any]:
    observations = {}
    for human_id in _human_ids(env):
        state = env._human_body.get_state(human_id)
        if state is not None:
            observations[human_id] = state.to_observation()
    return observations


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Unsupported teleop record value: {type(value).__name__}")


def _current_human_pose(
    env: IsaacAutonomousEnv,
    human_id: str,
) -> tuple[list[float], float]:
    state = _human_state(env, human_id)
    position = (
        state.get("position") or env._human_positions.get(human_id) or [0.0, 0.0, 0.0]
    )
    yaw = state.get("yaw_rad")
    if yaw is None:
        yaw = env._human_body.yaw(human_id)
    return [float(v) for v in position[:3]], float(yaw or 0.0)


def _current_human_z(env: IsaacAutonomousEnv, human_id: str) -> float:
    pos = env._human_positions.get(human_id)
    if pos is not None and len(pos) >= 3:
        return float(pos[2])
    return 0.0


def _pose_xy_yaw(env: IsaacAutonomousEnv) -> dict[str, Any]:
    pose = env._robot_pose()
    return {
        "position": pose.get("position") or [None, None, None],
        "yaw_rad": pose.get("yaw_rad"),
        "body_posture": pose.get("body_posture"),
        "camera_gaze_tilt_deg": pose.get("camera_gaze_tilt_deg"),
    }


def _step_simulation(
    env: IsaacAutonomousEnv,
    *,
    env_action: dict[str, Any],
    physics_ticks_per_sample: int,
) -> bool:
    action_name = (
        "move_body" if any(env_action.get("move_by_speed", [])[:3]) else "wait"
    )
    env._reject_crouched_g1_locomotion(action_name)
    decision = env._inject_g1_carry_pose(
        {"action": action_name, "controller_action": env_action}
    )
    result = env._robot_body.step_simulation(
        env=env._env,
        action_name=action_name,
        env_action=decision["controller_action"],
        physics_step_budget=int(physics_ticks_per_sample),
        global_tick=env._physics_steps,
        step_human_bodies=env._step_human_bodies,
        pump_kit_updates=env._pump_kit_updates,
        pump_motion_window=env._pump_human_motion_window,
        sample_body_dynamics=env._g1_body_dynamics,
        begin_decision_window=env._begin_decision_render_window,
        prepare_final_frame=env._prepare_final_decision_frame,
        end_decision_window=env._end_decision_render_window,
        diagnose_physics_tick=env._on_physics_tick,
        abort_requested=env._abort_requested,
    )
    if result.observation:
        env._obs = result.observation
    env._physics_steps += result.physics_steps
    return bool(result.terminated or result.truncated)


def _print_status(
    env: IsaacAutonomousEnv,
    session: TeleopSession,
    *,
    heading_yaw: float | None = None,
) -> None:
    pose = _pose_xy_yaw(env)
    print(
        f"[TELEOP] physics={env._physics_steps} active={session.active_agent} "
        f"human={session.active_human_id or '-'} viewer={session.viewer_mode} "
        f"robot={pose.get('position')} yaw={pose.get('yaw_rad')} "
        f"posture={pose.get('body_posture')} tilt={pose.get('camera_gaze_tilt_deg')}",
        flush=True,
    )
    snapshot = env.get_world_snapshot()
    for human_id, state in (snapshot.get("humans") or {}).items():
        marker = "*" if human_id == session.active_human_id else " "
        extra = ""
        if heading_yaw is not None and human_id == session.active_human_id:
            extra = f" cmd_yaw={heading_yaw:.2f}"
        print(
            f"  {marker}{human_id}: pos={state.get('position')} "
            f"yaw={state.get('yaw_rad')} action={state.get('active_action')} "
            f"status={state.get('action_status')} target={state.get('target_position')}"
            f"{extra}",
            flush=True,
        )


def _subscribe_keyboard() -> tuple[TeleopKeyboardState, dict[str, bool], Any]:
    """Subscribe to teleop keys after Kit starts; own only teleop input state."""
    import carb.input
    import omni.appwindow

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    interface = carb.input.acquire_input_interface()
    state = TeleopKeyboardState()
    pending = {"tab": False, "capture": False}
    motion_keys = {getattr(carb.input.KeyboardInput, name): name for name in "IKJLUO"}

    def on_key(input_event: Any) -> bool:
        event = input_event.event
        key = getattr(event, "input", None)
        kind = getattr(event, "type", None)
        if key in motion_keys:
            if kind in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                state.press(motion_keys[key])
            elif kind == carb.input.KeyboardEventType.KEY_RELEASE:
                state.release(motion_keys[key])
            return False
        if key in (carb.input.KeyboardInput.TAB, carb.input.KeyboardInput.F8):
            if kind == carb.input.KeyboardEventType.KEY_PRESS:
                pending["tab" if key == carb.input.KeyboardInput.TAB else "capture"] = (
                    True
                )
            return False
        return True

    # Run before Kit hotkeys: F8 also opens its profiler by default. Consume
    # owned events so one capture request cannot trigger both operations.
    subscription = interface.subscribe_to_input_events(on_key, device=keyboard, order=0)
    return state, pending, (interface, subscription)


def _unsubscribe_keyboard(subscription: Any) -> None:
    if subscription is not None:
        interface, token = subscription
        interface.unsubscribe_to_input_events(token)


def _grasp_step_record(info: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(info or {})
    skill = payload.get("grasp_skill")
    if not isinstance(skill, dict):
        skill = {}
    failure = payload.get("grasp_failure")
    if not isinstance(failure, dict):
        failure = {}
    return {
        "executed": bool(payload.get("executed")),
        "transport_ready": skill.get("transport_ready"),
        "object_id": skill.get("object_id"),
        "feedback": payload.get("feedback") or "",
        "grasp_failure_code": failure.get("code"),
        "grasp_failure": failure or None,
    }


def _execute_robot_tool(
    env: IsaacAutonomousEnv,
    tools: Any,
    *,
    name: str,
    args: dict[str, Any],
    action_step: int,
    image_dir: Path,
) -> tuple[Any, bool, dict[str, Any]]:
    context: dict[str, Any] = {"action_step": action_step}
    if name == "grasp_object":
        raw = env.observe()
        context["observation"] = agent_observation_from_runtime(
            raw,
            camera_views=env.agent_harness_config.protocol.camera_views,
        )
    call = ToolCall(
        tool=name,
        args=dict(args),
        call_id=f"teleop-{name}-{action_step}",
    )
    result = tools.execute(call, context)
    image_paths = []
    for index, data_url in enumerate(result.images):
        header, encoded = data_url.split(",", 1)
        extension = {"data:image/png;base64": "png", "data:image/jpeg;base64": "jpg"}[
            header
        ]
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{action_step:06d}_{name}_{index}.{extension}"
        path.write_bytes(base64.b64decode(encoded, validate=True))
        image_paths.append(str(path))
        print(f"[TELEOP] tool overlay: {path}", flush=True)
    evidence: dict[str, Any] = {"images": image_paths}
    print(f"[TELEOP] {name}: {result.content}", flush=True)
    if result.is_error:
        return result, False, evidence
    if result.action:
        _observation, done, info = env.step(result.action)
        del _observation
        feedback = info.get("feedback") if isinstance(info, dict) else None
        if feedback:
            print(f"[TELEOP] env.step feedback: {feedback}", flush=True)
        evidence["physical_result"] = info
        return result, bool(done), evidence
    return result, False, evidence


def _apply_stdin_command(
    command_line: str,
    *,
    env: IsaacAutonomousEnv,
    tools: Any,
    session: TeleopSession,
    default_human_id: str,
    action_step: int,
    image_dir: Path,
    capture_frame: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[bool, int, dict[str, Any]]:
    known = _human_ids(env)
    command = parse_command(
        command_line,
        known_human_ids=known,
        default_human_id=default_human_id or session.active_human_id,
    )
    record: dict[str, Any] = {
        "stdin_command": command.raw,
        "stdin_kind": command.kind,
    }
    if command.kind == "noop":
        return True, action_step, record
    if command.kind == "quit":
        print("[TELEOP] quit requested.", flush=True)
        return False, action_step, record
    if command.kind == "help":
        print(HELP, flush=True)
        return True, action_step, record
    if command.kind == "list":
        print(f"[TELEOP] humans={known}", flush=True)
        return True, action_step, record
    if command.kind == "status":
        _print_status(env, session)
        return True, action_step, record
    if command.kind == "switch_robot":
        session.switch_to_robot()
        print("[TELEOP] active_agent=robot", flush=True)
        record["active_agent"] = session.active_agent
        return True, action_step, record
    if command.kind == "switch_human":
        human_id = session.switch_to_human(
            command.payload.get("human_id"),
            known_human_ids=known,
            default_human_id=default_human_id or session.active_human_id,
        )
        print(f"[TELEOP] active_agent=human id={human_id}", flush=True)
        record["active_agent"] = session.active_agent
        record["active_human_id"] = human_id
        return True, action_step, record
    if command.kind == "ego":
        mode, apply_to_env, notice = resolve_ego_viewport(
            session,
            str(command.payload["viewer_mode"]),
        )
        if apply_to_env:
            env.set_viewer_mode(mode)
            print(f"[TELEOP] viewer={mode}", flush=True)
        else:
            print(f"[TELEOP] {notice}", flush=True)
        record["viewer_mode"] = mode
        if notice:
            record["viewer_notice"] = notice
        return True, action_step, record
    if command.kind == "capture":
        if capture_frame is None:
            raise RuntimeError("Static capture is not configured for this session.")
        record["capture"] = capture_frame(command.payload["label"])
        print(f"[TELEOP] capture: {record['capture']}", flush=True)
        return True, action_step, record
    if command.kind == "human_action":
        human_id = str(command.payload["human_id"])
        action = dict(command.payload["action"])
        if action.get("command") == "goto":
            target = list(action["target"])
            if target[2] is None:
                target[2] = _current_human_z(env, human_id)
            action["target"] = target
        result = env.apply_human_action(human_id, action)
        print(
            f"[TELEOP] {human_id}: executed={result.executed}; {result.feedback}",
            flush=True,
        )
        record["human_action"] = {
            "human_id": human_id,
            "action": action,
            "executed": result.executed,
            "feedback": result.feedback,
            "status": result.state.action_status if result.state is not None else None,
        }
        return True, action_step, record
    request = robot_tool_request(command)
    if request is None:
        raise ValueError(f"Unhandled command kind {command.kind!r}.")
    name, args = request
    if name == "grasp_object" and (
        not session.grasp_enabled or tools.handler(name) is None
    ):
        raise ValueError("Grasp requires launching teleop with --with-grasp.")
    result, done, evidence = _execute_robot_tool(
        env,
        tools,
        name=name,
        args=args,
        action_step=action_step,
        image_dir=image_dir,
    )
    action_step += 1
    record["tool"] = {
        "name": name,
        "args": args,
        "is_error": bool(result.is_error),
        "content": result.content,
        "metadata": dict(result.metadata or {}),
        "has_action": result.action is not None,
    }
    record.update(evidence)
    if name == "grasp_object":
        record["grasp_result"] = (
            _grasp_step_record(evidence["physical_result"])
            if result.action is not None and not result.is_error
            else {
                "executed": False,
                "error": result.content,
                "metadata": dict(result.metadata or {}),
            }
        )
    if done:
        print("[TELEOP] Env reported done after a harness skill.", flush=True)
        return False, action_step, record
    return True, action_step, record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively control the robot and animated NPCs in a TARSphere scene.",
    )
    parser.add_argument(
        "--episode",
        default=str(REPO_ROOT / "episodes/case/fetch_and_deliver/base.yaml"),
        help="Episode YAML. Fetch episodes need physical_task for --with-grasp.",
    )
    parser.add_argument(
        "--human-id",
        default="",
        help="Default human id for `human` and keyboard human loco.",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--viewer",
        choices=["free", "third_person", "first_person"],
        default="third_person",
        help="Initial viewport. `ego on/off` later switches first/third.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without an Isaac window. stdin skills still work. "
            "Keyboard I/K and viewport ego on/off are unavailable; "
            "ego sensors stay enabled for grasp observe()."
        ),
    )
    parser.add_argument(
        "--record-ego",
        action="store_true",
        help="Write ego RGB-D PNGs when the grasp skill observes the scene.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable Isaac Sim's native livestream transport.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
    )
    parser.add_argument("--physics-ticks-per-sample", type=int, default=8)
    parser.add_argument("--forward-scale", type=float, default=0.5)
    parser.add_argument("--strafe-scale", type=float, default=0.5)
    parser.add_argument("--yaw-scale", type=float, default=1.0)
    parser.add_argument("--stride-m", type=float, default=0.55)
    parser.add_argument("--yaw-rate-deg-s", type=float, default=90.0)
    parser.add_argument(
        "--heading-source",
        choices=["body", "world"],
        default="body",
    )
    parser.add_argument("--command-interval", type=float, default=0.35)
    parser.add_argument("--allow-replan-while-walking", action="store_true")
    parser.add_argument("--status-every", type=int, default=60)
    parser.add_argument("--max-loops", type=int, default=0)
    parser.add_argument(
        "--with-grasp",
        action="store_true",
        help="Enable the default visual grounding and physical grasp skill.",
    )
    parser.add_argument("--capture-width", type=int, default=2560)
    parser.add_argument("--capture-height", type=int, default=1920)
    parser.add_argument(
        "--capture-layers",
        action="store_true",
        help="Include native instance-segmented actor RGBA layers in static captures.",
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Record continuous robot-eye video."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.physics_ticks_per_sample <= 60:
        parser.error("--physics-ticks-per-sample must be between 1 and 60.")
    for name in (
        "forward_scale",
        "strafe_scale",
        "yaw_scale",
        "stride_m",
        "yaw_rate_deg_s",
        "command_interval",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.capture_width < 1 or args.capture_height < 1:
        parser.error("Capture dimensions must be positive.")
    if args.max_loops < 0 or args.status_every <= 0:
        parser.error("--max-loops must be nonnegative and --status-every positive.")

    from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator

    configure_cuda_allocator()
    from tarsphere.harness.agent import declared_episode_actions
    from tarsphere.harness.factory import build_robot_tool_registry
    from tarsphere.harness.harness import (
        AgentHarnessConfig,
    )
    from tarsphere.bench.case_loader import BenchmarkCaseLoader
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv, RenderMode

    episode = BenchmarkCaseLoader().load_file(args.episode)
    if not getattr(episode, "humans", None):
        raise RuntimeError(f"Episode has no humans: {args.episode}")
    if episode.runtime.max_decision_steps is not None:
        episode.runtime.max_decision_steps = max(
            int(episode.runtime.max_decision_steps), 10000
        )

    # Interactive control uses keyboard or move_body through env.step.
    # Navigation policy backends are unnecessary for operator-driven motion.
    harness_config = AgentHarnessConfig()
    actions = declared_episode_actions(episode)
    actions.difference_update(
        {"navigate_to_frontier", "navigate_to_pointgoal", "navigate_to_entity"}
    )
    if not args.with_grasp:
        actions.difference_update({"locate_object", "grasp_object"})
    tools, _memory = build_robot_tool_registry(
        agent_harness_config=harness_config,
        episode_actions=actions,
    )
    if args.with_grasp:
        require_grasp_stack(episode, tools)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = (
        args.output
        or Path.home() / ".local/share/tarsphere/demos/teleop" / stamp / "session.jsonl"
    )
    if out_path.exists():
        parser.error(f"Recording already exists: {out_path}; choose another --output.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = (
        str(args.image_dir or out_path.parent / "images") if args.record_ego else None
    )
    initial_viewer = args.viewer if args.viewer != "free" else "third_person"

    env = IsaacAutonomousEnv(
        episode,
        human_motion_backend="animation",
        render_mode=RenderMode.DEMO,
        enable_streaming=bool(args.stream),
        continuous_ego_video=args.record_video,
        continuous_ego_video_output_dir=str(out_path.parent / "video"),
        continuous_ego_video_slug="teleop",
        enable_human_ego_cameras=False,
        headless=bool(args.headless),
        image_dir=image_dir,
        viewer_mode=args.viewer,
        enable_ego_sensors=True,
        agent_harness_config=harness_config,
    )

    keyboard_subscription = None
    capture_recorder = None

    def capture_frame(label: str) -> dict[str, Any]:
        nonlocal capture_recorder
        if capture_recorder is None:
            from tarsphere.infra.perception.teleop_capture import TeleopCapture

            capture_recorder = TeleopCapture(
                env,
                out_path.parent / "captures",
                resolution=(args.capture_width, args.capture_height),
                actor_layers=args.capture_layers,
            )
        return capture_recorder.capture(label)

    try:
        print(
            "[TELEOP] Resetting Isaac scene. First reset may bake NavMesh.", flush=True
        )
        env.reset()
        env.configure_private_mapping_observer(
            observer=None, action_space=sorted(actions)
        )
        human_ids = _human_ids(env)
        default_human = args.human_id if args.human_id in human_ids else human_ids[0]
        if args.human_id and args.human_id not in human_ids:
            raise RuntimeError(
                f"Unknown --human-id {args.human_id!r}; valid={human_ids}"
            )
        session = TeleopSession(
            active_agent="robot",
            active_human_id=default_human,
            viewer_mode=initial_viewer if args.viewer != "free" else "third_person",
            headless=bool(args.headless),
            grasp_enabled=bool(args.with_grasp),
        )
        if args.viewer == "first_person" and not args.headless:
            env.set_viewer_mode("first_person")
            session.set_viewer_mode("first_person")

        keyboard = None
        key_pending = {"tab": False, "capture": False}
        if args.headless:
            print("[TELEOP] headless: keyboard is not subscribed.", flush=True)
        else:
            keyboard, key_pending, keyboard_subscription = _subscribe_keyboard()
        headings: dict[str, HumanKeyboardHeading] = {}
        position, body_yaw = _current_human_pose(env, session.active_human_id)
        heading_yaw = human_heading_from_body_yaw(body_yaw)
        headings[session.active_human_id] = HumanKeyboardHeading(heading_yaw)
        if args.headless:
            print(
                "[TELEOP] Ready (headless). No I/K and no viewport. "
                "Type help in this terminal for stdin commands.",
                flush=True,
            )
        else:
            print(
                "[TELEOP] Ready. Focus the Isaac window for I/K/J/L/U/O. "
                "Type help in this terminal for stdin commands.",
                flush=True,
            )
        print(
            f"[TELEOP] episode={args.episode} human={session.active_human_id} "
            f"viewer={args.viewer} headless={bool(args.headless)} "
            f"record_ego={bool(args.record_ego)} "
            f"grasp_object={tools.handler('grasp_object') is not None}",
            flush=True,
        )
        print(
            f"[TELEOP] robot scales forward={args.forward_scale} "
            f"strafe={args.strafe_scale} yaw={args.yaw_scale}; "
            f"human stride={args.stride_m} command_interval={args.command_interval}s",
            flush=True,
        )
        _print_status(env, session, heading_yaw=heading_yaw)

        loops = 0
        action_step = 0
        start = time.monotonic()
        last_command_times: dict[str, float] = {}
        last_targets: dict[str, list[float]] = {}
        stdin_open = True
        with out_path.open("x") as log_file:
            while env._env.simulation_app.is_running():
                loops += 1
                continue_run = True
                stdin_records: list[dict[str, Any]] = []
                while stdin_open:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if not ready:
                        break
                    line = sys.stdin.readline()
                    if line == "":
                        stdin_open = False
                        if args.headless:
                            print(
                                "[TELEOP] stdin closed; stopping headless teleop.",
                                flush=True,
                            )
                            continue_run = False
                        else:
                            print(
                                "[TELEOP] stdin closed; simulation continues until "
                                "max_loops/window close.",
                                flush=True,
                            )
                        break
                    command_error = None
                    physics_start = env._physics_steps
                    try:
                        if keyboard is not None:
                            keyboard.clear()
                        continue_run, action_step, stdin_record = _apply_stdin_command(
                            line,
                            env=env,
                            tools=tools,
                            session=session,
                            default_human_id=default_human,
                            action_step=action_step,
                            image_dir=out_path.parent / "tool_images",
                            capture_frame=capture_frame,
                        )
                    except Exception as exc:
                        command_error = exc
                        print(f"[TELEOP] command error: {exc}", flush=True)
                        stdin_record = {
                            "stdin_command": line.strip(),
                            "error": str(exc),
                        }
                    finally:
                        if keyboard is not None:
                            keyboard.clear()
                    log_file.write(
                        json.dumps(
                            {
                                "loop": loops,
                                "physics_start": physics_start,
                                "physics_steps": env._physics_steps,
                                "simulation_time_s": env._physics_steps
                                / SIMULATION_PHYSICS_HZ,
                                **stdin_record,
                            },
                            default=_json_default,
                        )
                        + "\n"
                    )
                    log_file.flush()
                    if command_error is not None and not isinstance(
                        command_error, (ValueError, ActionRefusedError)
                    ):
                        raise command_error
                    if not continue_run:
                        break
                if not continue_run:
                    break

                if key_pending["tab"]:
                    key_pending["tab"] = False
                    keyboard.clear()
                    if session.active_agent == "robot":
                        session.switch_to_human(
                            session.active_human_id,
                            known_human_ids=_human_ids(env),
                            default_human_id=default_human,
                        )
                    else:
                        session.switch_to_robot()
                    print(
                        f"[TELEOP] Tab -> active_agent={session.active_agent}",
                        flush=True,
                    )
                    stdin_records.append(
                        {
                            "stdin_command": "tab",
                            "active_agent": session.active_agent,
                        }
                    )

                if key_pending["capture"]:
                    key_pending["capture"] = False
                    keyboard.clear()
                    try:
                        metadata = capture_frame("frame")
                        stdin_records.append(
                            {"stdin_command": "F8", "capture": metadata}
                        )
                    finally:
                        keyboard.clear()
                now = env._physics_steps / SIMULATION_PHYSICS_HZ
                command = (
                    IdleKeyboardCommand() if keyboard is None else keyboard.sample(now)
                )
                dt = args.physics_ticks_per_sample / SIMULATION_PHYSICS_HZ
                motion_refusal = None
                speed = [0.0, 0.0, 0.0]
                injected_action: dict[str, Any] | None = None
                if session.active_agent == "robot":
                    speed = speed_from_keyboard(
                        command,
                        forward_scale=args.forward_scale,
                        strafe_scale=args.strafe_scale,
                        yaw_scale=args.yaw_scale,
                    )
                    try:
                        terminated = _step_simulation(
                            env,
                            env_action={"move_by_speed": speed},
                            physics_ticks_per_sample=int(args.physics_ticks_per_sample),
                        )
                    except ActionRefusedError as refusal:
                        motion_refusal = refusal.as_feedback()
                        print(f"[TELEOP] {refusal}", flush=True)
                        if keyboard is not None:
                            keyboard.clear()
                        speed = [0.0, 0.0, 0.0]
                        terminated = _step_simulation(
                            env,
                            env_action={"move_by_speed": speed},
                            physics_ticks_per_sample=int(args.physics_ticks_per_sample),
                        )
                else:
                    forward, lateral, yaw_cmd = keyboard_vector(command)
                    position, body_yaw = _current_human_pose(
                        env, session.active_human_id
                    )
                    heading = headings.setdefault(
                        session.active_human_id,
                        HumanKeyboardHeading(human_heading_from_body_yaw(body_yaw)),
                    )
                    heading_yaw = heading.update(
                        body_yaw=body_yaw,
                        yaw_delta=math.radians(float(args.yaw_rate_deg_s))
                        * yaw_cmd
                        * dt,
                        follow_body=args.heading_source == "body",
                    )
                    state_before = _human_state(env, session.active_human_id)
                    if (
                        (forward or lateral)
                        and now - last_command_times.get(session.active_human_id, 0.0)
                        >= float(args.command_interval)
                        and can_inject_human_step(
                            state_before,
                            allow_replan_while_walking=bool(
                                args.allow_replan_while_walking
                            ),
                        )
                    ):
                        target = target_from_command(
                            position,
                            heading_yaw,
                            forward=forward,
                            lateral=lateral,
                            stride_m=float(args.stride_m),
                        )
                        injected_action = {
                            "command": "goto",
                            "target": target,
                            "end_rot": human_end_rotation_deg(
                                human_movement_heading(
                                    heading_yaw, forward=forward, lateral=lateral
                                )
                            ),
                        }
                        result = env.apply_human_action(
                            session.active_human_id, injected_action
                        )
                        last_command_times[session.active_human_id] = now
                        if result.executed:
                            heading.commit_goto()
                            last_targets[session.active_human_id] = target
                        stdin_records.append(
                            {
                                "human_id": session.active_human_id,
                                "action": injected_action,
                                "executed": bool(result.executed),
                                "feedback": result.feedback,
                            }
                        )
                        if not result.executed:
                            print(
                                f"[TELEOP] human motion refused: {result.feedback}",
                                flush=True,
                            )
                    terminated = _step_simulation(
                        env,
                        env_action={"move_by_speed": [0.0, 0.0, 0.0]},
                        physics_ticks_per_sample=int(args.physics_ticks_per_sample),
                    )

                pose = _pose_xy_yaw(env)
                record = {
                    "loop": loops,
                    "elapsed_s": round(time.monotonic() - start, 3),
                    "physics_steps": env._physics_steps,
                    "simulation_time_s": env._physics_steps / SIMULATION_PHYSICS_HZ,
                    "active_agent": session.active_agent,
                    "active_human_id": session.active_human_id,
                    "viewer_mode": session.viewer_mode,
                    "keyboard": [float(v) for v in command],
                    "motion_refusal": motion_refusal,
                    "move_by_speed": speed,
                    "human_injected_action": injected_action,
                    "humans": _human_observations(env),
                    "last_human_target": last_targets.get(session.active_human_id),
                    "pose": pose,
                    "stdin": stdin_records,
                }
                log_file.write(json.dumps(record, default=_json_default) + "\n")
                log_file.flush()

                if injected_action is not None:
                    print(
                        f"[TELEOP] GoTo {session.active_human_id} -> "
                        f"{[round(v, 3) for v in injected_action['target']]} "
                        f"end_rot={injected_action['end_rot']}",
                        flush=True,
                    )
                if loops % max(1, int(args.status_every)) == 0:
                    _print_status(env, session, heading_yaw=heading_yaw)
                if terminated:
                    print("[TELEOP] Env terminated/truncated; stopping.", flush=True)
                    break
                if args.max_loops and loops >= int(args.max_loops):
                    print(
                        f"[TELEOP] max_loops={args.max_loops} reached; stopping.",
                        flush=True,
                    )
                    break
    finally:
        try:
            _unsubscribe_keyboard(keyboard_subscription)
            if capture_recorder is not None:
                capture_recorder.close()
        finally:
            try:
                video = env.finish_continuous_ego_video()
                if video is not None:
                    print(
                        f"[TELEOP] Video written to {out_path.parent / 'video'}",
                        flush=True,
                    )
            finally:
                env.close(skip_kit_cleanup=True)
        print(f"[TELEOP] Log written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
