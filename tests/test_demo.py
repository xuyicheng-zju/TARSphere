"""Public demo selection, capture isolation, and evidence verification."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tarsphere/cli/demo.py"
spec = importlib.util.spec_from_file_location("demo_launch", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.mark.parametrize(
    "line",
    [
        "move nan 0 0",
        "wait -1",
        "tilt 90",
        "crouch 0.1",
        "look Nora 0",
        "sit Nora /Chair -1",
        "goto Unknown 1 2",
    ],
)
def test_teleop_rejects_invalid_motion_before_execution(line):
    from tarsphere.infra.control.teleop import parse_command

    with pytest.raises(ValueError):
        parse_command(line, known_human_ids=["Nora"])


def test_teleop_motion_routes_through_existing_tools():
    from tarsphere.infra.control.teleop import (
        parse_command,
        robot_tool_request,
    )

    move = parse_command("move 0.3 -0.2 15")
    assert robot_tool_request(move) == (
        "move_body",
        {"delta": [0.3, -0.2, 15.0]},
    )
    assert robot_tool_request(parse_command("wait 0.5")) == (
        "wait",
        {"seconds": 0.5},
    )


@pytest.mark.parametrize("executed", [False, True])
def test_teleop_direct_grasp_observes_once_and_records_physical_feedback(
    monkeypatch, tmp_path, executed
):
    from tarsphere.cli import teleop
    from tarsphere.infra.control.teleop import TeleopSession

    observation = object()
    captures, contexts, calls, actions = [], [], [], []
    physical = {
        "executed": executed,
        "feedback": "Object retained" if executed else "Target slipped",
        "grasp_skill": {"transport_ready": executed, "object_id": "cup"},
    }

    def observe():
        captures.append(observation)
        return observation

    def execute(call, context):
        calls.append(call)
        contexts.append(context["observation"])
        return SimpleNamespace(
            is_error=False,
            content="Grasp command admitted",
            action={"action": "grasp_object", "target": "cup"},
            metadata={},
            images=(),
        )

    def step(action):
        actions.append(action)
        return None, False, physical

    monkeypatch.setattr(
        teleop, "agent_observation_from_runtime", lambda raw, **kwargs: raw
    )
    env = SimpleNamespace(
        observe=observe,
        step=step,
        _human_positions={},
        agent_harness_config=SimpleNamespace(
            protocol=SimpleNamespace(camera_views=("ego",))
        ),
    )
    session = TeleopSession(grasp_enabled=True)
    tools = SimpleNamespace(execute=execute, handler=lambda name: execute)
    keep_running, action_step, record = teleop._apply_stdin_command(
        "grasp cup",
        env=env,
        tools=tools,
        session=session,
        default_human_id="",
        action_step=0,
        image_dir=tmp_path,
    )
    assert captures == [observation]
    assert contexts == [observation]
    assert len(calls) == 1 and calls[0].args == {"target": "cup"}
    assert actions == [{"action": "grasp_object", "target": "cup"}]
    assert keep_running and action_step == 1
    assert record["physical_result"] == physical
    assert record["grasp_result"]["executed"] is executed
    assert record["grasp_result"]["transport_ready"] is executed
    assert record["grasp_result"]["feedback"] == physical["feedback"]


@pytest.mark.parametrize("speed", [[0.2, 0, 0], [0, 0, 0]])
def test_teleop_keyboard_ticks_preserve_carry_and_human_physics_clock(speed):
    from tarsphere.cli.teleop import _step_simulation
    from tarsphere.embodiment.dex3 import dex3_controller_name

    calls, decisions, posture_checks = [], [], []
    held_upper_body = [0.2] * 15
    held_fingers = [[0.4] * 7]
    finger_controller = dex3_controller_name("right")

    def inject_carry(decision):
        decisions.append(decision)
        return {
            **decision,
            "controller_action": {
                "move_by_speed": [*speed, 0.74, held_upper_body],
                finger_controller: held_fingers,
            },
        }

    def step_simulation(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            observation={"frame": "current"},
            physics_steps=8,
            terminated=False,
            truncated=False,
        )

    env = SimpleNamespace(
        _robot_body=SimpleNamespace(step_simulation=step_simulation),
        _env=object(),
        _physics_steps=1200,
        _step_human_bodies=lambda *args: None,
        _pump_kit_updates=lambda: None,
        _pump_human_motion_window=lambda *args: None,
        _g1_body_dynamics=lambda: None,
        _begin_decision_render_window=lambda *args: None,
        _prepare_final_decision_frame=lambda: None,
        _end_decision_render_window=lambda: None,
        _on_physics_tick=lambda *args: None,
        _abort_requested=lambda: False,
        _reject_crouched_g1_locomotion=posture_checks.append,
        _inject_g1_carry_pose=inject_carry,
    )
    assert not _step_simulation(
        env, env_action={"move_by_speed": speed}, physics_ticks_per_sample=8
    )
    action = "move_body" if any(speed) else "wait"
    assert posture_checks == [action]
    assert decisions == [
        {"action": action, "controller_action": {"move_by_speed": speed}}
    ]
    assert calls[0]["env_action"]["move_by_speed"][4] is held_upper_body
    assert calls[0]["env_action"][finger_controller] is held_fingers
    assert calls[0]["global_tick"] == 1200
    assert calls[0]["diagnose_physics_tick"] is env._on_physics_tick
    assert calls[0]["pump_motion_window"] is env._pump_human_motion_window
    assert env._physics_steps == 1208
    assert env._obs == {"frame": "current"}


def test_teleop_rejects_crouched_keyboard_motion_before_any_physics():
    from tarsphere.cli.teleop import _step_simulation

    def reject(action):
        assert action == "move_body"
        raise ValueError("Stand before walking")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Rejected crouch motion must not advance or inject controls"
        )

    env = SimpleNamespace(
        _reject_crouched_g1_locomotion=reject,
        _inject_g1_carry_pose=forbidden,
        _robot_body=SimpleNamespace(step_simulation=forbidden),
        _physics_steps=23,
    )
    with pytest.raises(ValueError, match="Stand before walking"):
        _step_simulation(
            env, env_action={"move_by_speed": [0.2, 0, 0]}, physics_ticks_per_sample=8
        )
    assert env._physics_steps == 23


def test_teleop_logs_native_human_status_and_numpy_pose_values():
    import numpy as np
    from tarsphere.cli.teleop import _human_observations, _json_default
    from tarsphere.infra.control.human_body import HumanBodyState

    state = HumanBodyState(
        human_id="Nora",
        position=[np.float32(1), np.float32(2), np.float32(0)],
        active_action="goto",
        action_status="ongoing",
    )
    env = SimpleNamespace(
        _human_positions={"Nora": state.position},
        _human_body=SimpleNamespace(get_state=lambda name: state),
    )
    recorded = json.loads(json.dumps(_human_observations(env), default=_json_default))
    assert recorded["Nora"]["position"] == [1, 2, 0]
    assert recorded["Nora"]["action_status"] == "ongoing"
    state.action_status = "completed"
    assert _human_observations(env)["Nora"]["action_status"] == "completed"


def test_teleop_preserves_rejected_human_action_feedback(tmp_path):
    from tarsphere.cli.teleop import _apply_stdin_command
    from tarsphere.infra.control.teleop import TeleopSession

    env = SimpleNamespace(
        _human_positions={"Nora": [0, 0, 0]},
        apply_human_action=lambda *args: SimpleNamespace(
            executed=False, feedback="Seat missing", state=None
        ),
    )
    keep_running, _, record = _apply_stdin_command(
        "sit Nora /Chair",
        env=env,
        tools=None,
        session=TeleopSession(),
        default_human_id="Nora",
        action_step=0,
        image_dir=tmp_path,
    )
    assert keep_running
    assert record["human_action"]["executed"] is False
    assert record["human_action"]["feedback"] == "Seat missing"


def test_teleop_capture_records_callback_without_advancing_physics(tmp_path):
    from tarsphere.cli.teleop import _apply_stdin_command
    from tarsphere.infra.control.teleop import TeleopSession

    captures = []
    artifact = {"path": str(tmp_path / "frame.png"), "physics_step": 42}

    def capture(label):
        captures.append(label)
        return artifact

    def forbidden(*args, **kwargs):
        raise AssertionError("A still capture must not execute a physical action")

    env = SimpleNamespace(_human_positions={}, _physics_steps=42, step=forbidden)
    keep_running, action_step, record = _apply_stdin_command(
        "capture front_view",
        env=env,
        tools=SimpleNamespace(execute=forbidden),
        session=TeleopSession(),
        default_human_id="",
        action_step=7,
        image_dir=tmp_path,
        capture_frame=capture,
    )
    assert keep_running and action_step == 7
    assert captures == ["front_view"]
    assert record["stdin_kind"] == "capture"
    assert record["capture"] is artifact
    assert env._physics_steps == 42


def test_teleop_keyboard_adapter_routes_events_and_unsubscribes(monkeypatch):
    from tarsphere.cli.teleop import _subscribe_keyboard, _unsubscribe_keyboard
    from tarsphere.infra.control.teleop import keyboard_vector

    subscriptions, unsubscribed = [], []
    keyboard = object()
    token = object()

    def subscribe(callback, *, device, order):
        assert device is keyboard
        assert order == 0
        subscriptions.append(callback)
        return token

    interface = SimpleNamespace(
        subscribe_to_input_events=subscribe,
        unsubscribe_to_input_events=lambda *args: unsubscribed.append(args),
    )
    inputs = ModuleType("carb.input")
    inputs.KeyboardInput = SimpleNamespace(
        **{key: key for key in [*"IKJLUO", "TAB", "F8"]}
    )
    inputs.KeyboardEventType = SimpleNamespace(
        KEY_PRESS="press", KEY_REPEAT="repeat", KEY_RELEASE="release"
    )
    inputs.acquire_input_interface = lambda: interface
    carb = ModuleType("carb")
    carb.input = inputs
    appwindow = ModuleType("omni.appwindow")
    appwindow.get_default_app_window = lambda: SimpleNamespace(
        get_keyboard=lambda: keyboard
    )
    omni = ModuleType("omni")
    omni.appwindow = appwindow
    for module in (carb, inputs, omni, appwindow):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    state, pending, subscription = _subscribe_keyboard()

    def callback(event):
        return subscriptions[0](SimpleNamespace(event=event))

    for key in "IU":
        assert callback(SimpleNamespace(input=key, type="press")) is False
    callback(SimpleNamespace(input="U", type="repeat"))
    callback(SimpleNamespace(input="U", type="release"))
    assert keyboard_vector(state.sample(10)) == (1, 0, 1)
    callback(SimpleNamespace(input="F8", type="release"))
    assert keyboard_vector(state.sample(10.2)) == (1, 0, 0)
    for key in ("TAB", "F8"):
        assert callback(SimpleNamespace(input=key, type="press")) is False
        assert callback(SimpleNamespace(input=key, type="repeat")) is False
        assert callback(SimpleNamespace(input=key, type="release")) is False
    assert pending == {"tab": True, "capture": True}
    assert callback(SimpleNamespace(input="ESCAPE", type="press")) is True
    _unsubscribe_keyboard(subscription)
    assert unsubscribed == [(token,)]


@pytest.mark.parametrize(
    "episode_arguments,expected_limit",
    [
        ([], 10000),
        (["--episode", str(demo.ROOT / "examples/episodes/navigation.yaml")], None),
    ],
)
def test_headless_teleop_quit_is_recorded_without_model_services(
    monkeypatch, tmp_path, episode_arguments, expected_limit
):
    from tarsphere.cli import teleop
    from tarsphere.harness import factory

    instances, registries = [], []

    class FakeEnv:
        def __init__(self, episode, **kwargs):
            self.episode_limit = episode.runtime.max_decision_steps
            self.options = kwargs
            self._human_positions = {"Nora": [0, 0, 0]}
            self._physics_steps = 0
            self._env = SimpleNamespace(
                simulation_app=SimpleNamespace(is_running=lambda: True)
            )
            instances.append(self)

        def reset(self):
            return {}

        def configure_private_mapping_observer(self, *, observer, action_space):
            assert observer is None
            self.mapping_actions = set(action_space)

        def _robot_pose(self):
            return {"position": [0, 0, 0], "yaw_rad": 0}

        def get_world_snapshot(self, *args):
            return {"humans": {"Nora": {"position": [0, 0, 0], "yaw_rad": 0}}}

        def close(self, **kwargs):
            self.closed = True

        def finish_continuous_ego_video(self):
            self.video_finalized = True
            return None

    def build_tools(**kwargs):
        registries.append(kwargs)
        return SimpleNamespace(handler=lambda name: None), None

    simulator = ModuleType("tarsphere.infra.simulation.isaac_env")
    simulator.IsaacAutonomousEnv = FakeEnv
    simulator.RenderMode = SimpleNamespace(DEMO="demo")
    monkeypatch.setitem(sys.modules, simulator.__name__, simulator)
    monkeypatch.setattr(factory, "build_robot_tool_registry", build_tools)
    monkeypatch.setattr(sys, "stdin", io.StringIO("quit\n"))
    monkeypatch.setattr(teleop.select, "select", lambda *args: ([sys.stdin], [], []))
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log = tmp_path / "session.jsonl"
    teleop.main(
        [
            "--headless",
            "--stream",
            "--record-video",
            "--output",
            str(log),
            *episode_arguments,
        ]
    )
    record = json.loads(log.read_text())
    assert record["stdin_kind"] == "quit"
    assert record["physics_start"] == record["physics_steps"] == 0
    assert record["simulation_time_s"] == 0
    assert len(instances) == len(registries) == 1
    assert instances[0].options["human_motion_backend"] == "animation"
    assert instances[0].options["continuous_ego_video"] is True
    assert instances[0].options["enable_streaming"] is True
    assert instances[0].episode_limit == expected_limit
    assert not instances[0].mapping_actions.intersection(
        {"navigate_to_frontier", "navigate_to_pointgoal", "navigate_to_entity"}
    )
    assert "move_body" in instances[0].mapping_actions
    assert instances[0].closed
    assert instances[0].video_finalized


@pytest.mark.parametrize(
    "launcher", [["scripts/teleop.sh"], ["scripts/demo.sh", "teleop"]]
)
def test_teleop_help_runs_without_activating_the_simulator(launcher):
    import os

    env = dict(os.environ, TARSPHERE_PYTHON=sys.executable)
    result = subprocess.run(
        ["bash", *launcher, "--help"],
        cwd=demo.ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--with-grasp" in result.stdout
    assert "--record-video" in result.stdout


def _write_default_demo_config(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": demo.default_cases()}))
    return path


def test_real_manifest_covers_both_families_and_all_six_conditions():
    cases = demo.load_cases()
    assert len(cases) == 12
    assert {c["id"] for c in cases} == {
        f"{family}-{condition}"
        for family in ("navigate_and_report", "fetch_and_deliver")
        for condition in ("base", "pr", "da", "tu", "hc", "iu")
    }
    assert len({c["episode_id"] for c in cases}) == 12
    assert len({c["scene_name"] for c in cases}) == 1
    for group in ("navigate_and_report", "fetch_and_deliver"):
        import yaml

        episodes = [
            yaml.safe_load((demo.ROOT / case["episode"]).read_text())
            for case in cases
            if case["id"].startswith(group + "-")
        ]
        assert all(episode["robot"] == episodes[0]["robot"] for episode in episodes)
        assert all(episode["scene"] == episodes[0]["scene"] for episode in episodes)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("empty", "nonempty cases list"),
        ("duplicate", "duplicate demo slot"),
        ("binding", "wrong task family or condition"),
    ],
)
def test_empty_duplicate_or_wrong_condition_is_rejected(tmp_path, mutation, message):
    data = {"cases": demo.default_cases()}
    if mutation == "empty":
        data["cases"].clear()
    elif mutation == "duplicate":
        data["cases"][-1] = data["cases"][0]
    else:
        data["cases"][1]["episode"] = data["cases"][0]["episode"]
    config = tmp_path / "cases.json"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        demo.load_cases(config)


def test_selected_highlights_group_by_scene_without_repeating_other_slots(tmp_path):
    import yaml

    selected = [
        ("navigate_and_report-base", "navigate_and_report", "base", "open_lounge_home"),
        ("fetch_and_deliver-tu", "fetch_and_deliver", "task_update", "courtyard_home"),
        (
            "navigate_and_report-iu",
            "navigate_and_report",
            "intent_understanding",
            "courtyard_home",
        ),
    ]
    cases = []
    for key, family, condition, scene in selected:
        episode = yaml.safe_load(
            (demo.ROOT / f"episodes/case/{family}/{condition}.yaml").read_text()
        )
        episode["scene"]["asset_path"] = f"scenes/{scene}/scene.usd"
        path = tmp_path / scene / f"{key}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(episode))
        cases.append({"id": key, "episode": str(path)})
    config = tmp_path / "highlights.json"
    config.write_text(json.dumps({"cases": cases}))
    plan = json.loads(
        subprocess.run(
            [
                demo.sys.executable,
                str(SCRIPT),
                "plan",
                "--config",
                str(config),
                "--ego-only",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert plan["selected"] == [
        "navigate_and_report-base",
        "fetch_and_deliver-tu",
        "navigate_and_report-iu",
    ]
    assert len(plan["commands"]) == 2
    scenes = []
    for command in plan["commands"]:
        episodes = [
            Path(command[i + 1])
            for i, value in enumerate(command)
            if value == "--episode"
        ]
        assert len({p.parent.name for p in episodes}) == 1
        scenes.extend(p.parent.name for p in episodes)
        assert command.count("--room-observer-view") == 1
        assert command[command.index("--human-motion-backend") + 1] == "animation"
        assert command[command.index("--max-steps") + 1] == "100"
    assert sorted(scenes) == ["courtyard_home", "courtyard_home", "open_lounge_home"]
    assert plan["contract"]["capture"]["model_observation_resolution"] == [640, 480]
    assert plan["contract"]["capture"]["presentation_ego_resolution"] == [2560, 1920]


def test_same_scene_name_cannot_merge_different_scene_assets(monkeypatch, tmp_path):
    import yaml

    original = yaml.safe_load(
        (demo.ROOT / {"cases": demo.default_cases()}["cases"][0]["episode"]).read_text()
    )
    items = []
    for i, family in enumerate(("navigate_and_report", "fetch_and_deliver")):
        path = tmp_path / str(i) / "same_scene" / "episode.yaml"
        path.parent.mkdir(parents=True)
        episode = yaml.safe_load(yaml.safe_dump(original))
        episode["scene"]["asset_path"] = f"assets/scene-{i}.usd"
        episode["evaluation"]["task_profile"]["mainline"] = demo.FAMILIES[family]
        path.write_text(yaml.safe_dump(episode))
        items.append(
            {"id": family + "-base", "episode": str(path.relative_to(tmp_path))}
        )
    config = tmp_path / "cases.json"
    config.write_text(json.dumps({"cases": items}))
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    with pytest.raises(
        ValueError, match="different scene assets would share an output directory"
    ):
        demo.load_cases(config)


def test_demo_environment_selects_physical_gpu_and_cannot_reuse_formal_lane(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("TARSPHERE_DEMO_GPU_UUID", "stale-identity")
    monkeypatch.setenv("TARSPHERE_ISAAC_RUNTIME_SLOT", "3")
    monkeypatch.setenv("TARSPHERE_KEEP_WORKER_ALIVE", "1")
    monkeypatch.setenv("TARSPHERE_NAVMESH_USE_GPU", "1")
    env = demo.capture_environment(2, 19990)
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "TARSPHERE_DEMO_GPU_UUID" not in env
    assert env["TARSPHERE_KIT_ACTIVE_GPU"] == "2"
    assert env["TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES"] == "2"
    assert env["TARSPHERE_FRONTIERNET_DEVICE"] == "cuda:2"
    assert env["TARSPHERE_ISAAC_RUNTIME_SLOT"] == "0"
    assert env["TARSPHERE_XNAVDP_PORT"] == "19990"
    assert "TARSPHERE_KEEP_WORKER_ALIVE" not in env
    assert env["TARSPHERE_PRESENTATION_CAPTURE"] == "1"
    assert env["TARSPHERE_NAVMESH_USE_GPU"] == "0"
    assert demo.CAPTURE["navmesh_bake_backend"] == "native_cpu"
    assert env["TARSPHERE_ROBOT_CAMERA_WIDTH"] == "640"
    assert env["TARSPHERE_ROBOT_CAMERA_HEIGHT"] == "480"
    assert demo.CAPTURE["model_observation_resolution"] == [640, 480]
    assert demo.CAPTURE["presentation_ego_resolution"] == [2560, 1920]


def test_isolated_demo_environment_uses_one_verified_uuid_and_physical_lock_index(
    monkeypatch,
):
    from tarsphere.infra.simulation import demo_gpu

    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000004"
    queries = []

    def nvml_run(command, **kwargs):
        queries.append(command)
        assert "--id=4" in command
        return subprocess.CompletedProcess(command, 0, f"4, {gpu_uuid}\n", "")

    monkeypatch.setattr(demo_gpu.subprocess, "run", nvml_run)
    env = demo.capture_environment(4, 19990, isolated_gpu=True, runtime_slot=2)
    assert len(queries) == 2  # Lookup and independent identity validation.
    assert env["TARSPHERE_KIT_ACTIVE_GPU"] == "4"
    assert env["TARSPHERE_ISAAC_RUNTIME_SLOT"] == "2"
    assert env["TARSPHERE_DEMO_GPU_UUID"] == env["CUDA_VISIBLE_DEVICES"] == gpu_uuid
    assert env["TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES"] == gpu_uuid
    assert env["TARSPHERE_MODEL_GPU"] == "0"
    assert env["TARSPHERE_FRONTIERNET_DEVICE"] == "cuda:0"
    assert env["TARSPHERE_NAVMESH_USE_GPU"] == "1"
    assert env["TARSPHERE_XNAVDP_PORT"] == "19990"
    assert env["TARSPHERE_ROBOT_CAMERA_WIDTH"] == "640"
    assert demo.CAPTURE["navmesh_bake_backend"] == "native_cpu"


def test_isolated_environment_rejects_identity_change_before_worker_launch(monkeypatch):
    from tarsphere.infra.simulation import demo_gpu

    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000004"
    returned = iter([gpu_uuid, "GPU-00000000-0000-0000-0000-000000000005"])
    monkeypatch.setattr(
        demo_gpu, "lookup_physical_gpu_uuid", lambda _gpu: next(returned)
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        demo.capture_environment(4, 19990, isolated_gpu=True)


def test_plan_is_offline_and_prints_only_selected_commands(tmp_path):
    result = subprocess.run(
        [demo.sys.executable, str(SCRIPT), "plan", "--case", "fetch_and_deliver-tu"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["selected"] == ["fetch_and_deliver-tu"]
    assert len(plan["contract"]["cases"]) == 12
    command = plan["commands"][0]
    assert "--record-video" in command
    assert "--room-observer-max-frames" not in command
    assert "--api-key" not in command
    assert "--agent-mode" not in command
    assert command[command.index("--render-mode") + 1] == "demo"
    assert command[command.index("--human-motion-backend") + 1] == "animation"
    assert command[command.index("--max-steps") + 1] == str(
        demo.FORMAL_EVALUATION_MAX_DECISION_STEPS
    )
    assert (
        plan["contract"]["capture"]["max_model_decisions"]
        == demo.FORMAL_EVALUATION_MAX_DECISION_STEPS
    )
    assert "presentation_ego" in command


def test_demo_worker_options_exist_in_current_worker_cli(tmp_path):
    import ast

    options = set()
    for path in (
        demo.ROOT / "tarsphere/cli/worker.py",
        demo.ROOT / "tarsphere/harness/harness_cli.py",
    ):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                options.update(
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("--")
                )
    cases = demo.load_cases()[:1]
    for capture in (demo.CAPTURE, demo.DEBUG_CAPTURE):
        command = demo.worker_command(cases, tmp_path, "test-model", capture=capture)
        assert {arg for arg in command if arg.startswith("--")} <= options


def test_debug_plan_keeps_policy_640_and_uses_distinct_low_cost_capture_contract(
    tmp_path,
):
    result = subprocess.run(
        [
            demo.sys.executable,
            str(SCRIPT),
            "plan",
            "--debug",
            "--case",
            "navigate_and_report-tu",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    capture = plan["contract"]["capture"]
    assert (
        capture["ego_resolution"]
        == capture["model_observation_resolution"]
        == [640, 480]
    )
    assert capture["presentation_ego_resolution"] == [640, 480]
    assert capture["observer_resolution"] == [640, 360]
    assert capture["purpose"].startswith("debug_only;")
    command = plan["commands"][0]
    assert command[command.index("--room-observer-width") + 1] == "640"
    assert command[command.index("--room-observer-height") + 1] == "360"
    assert command[command.index("--max-steps") + 1] == "100"
    assert demo.CAPTURE["observer_resolution"] == [2560, 1440]


@pytest.mark.parametrize(
    "debug, resolution", [(True, [640, 480]), (False, [2560, 1920])]
)
def test_ego_only_plan_omits_all_third_person_products(tmp_path, debug, resolution):
    command = [
        demo.sys.executable,
        str(SCRIPT),
        "plan",
        "--ego-only",
        "--config",
        str(_write_default_demo_config(tmp_path)),
        "--case",
        "navigate_and_report-base",
        "--case",
        "fetch_and_deliver-base",
    ]
    if debug:
        command.append("--debug")
    plan = json.loads(
        subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
    )
    capture = plan["contract"]["capture"]
    assert capture["observer_views"] == ["presentation_ego"]
    assert capture["presentation_ego_resolution"] == resolution
    assert capture["model_observation_resolution"] == [640, 480]
    worker = plan["commands"][0]
    assert worker.count("--episode") == 2
    assert worker.count("--room-observer-view") == 1
    assert worker[worker.index("--room-observer-view") + 1] == "presentation_ego"
    assert "follow" not in worker
    assert demo.CAPTURE["observer_views"] == ["presentation_ego", "follow"]


def test_ego_only_capture_verification_does_not_require_missing_follow(
    monkeypatch, tmp_path
):
    case = {"id": "navigate_and_report-base", "episode_id": "episode"}
    movie = tmp_path / "ego.mp4"
    movie.write_bytes(b"verified native source")
    frames = tmp_path / "episode_ego_frames.jsonl"
    frames.write_text(json.dumps({"physics_step": 8, "video_physics_step": 8}) + "\n")
    result = {
        "episode_id": "episode",
        "success": False,
        "agent_model_usage": {"by_model": {"gpt-6.1-sol": {}}},
        "continuous_ego_videos": {"ego": str(movie), "frame_manifest": str(frames)},
        "observer_videos": {
            "presentation_ego": {"path": str(movie), "frame_manifest": str(frames)}
        },
    }
    for key in [
        "heartbeat_path",
        "model_output_trace_path",
        "rollout_artifact_path",
        "episode_artifact_path",
    ]:
        artifact = tmp_path / key
        artifact.write_text("{}")
        result[key] = str(artifact)
    (tmp_path / "episode.json").write_text(json.dumps(result))
    inputs = tmp_path / "episode.model_output.inputs"
    inputs.mkdir()
    (inputs / "001.request.json").write_text("{}")
    for folder, name in [("rgbd", "001.npz"), ("memory", "001.json")]:
        path = tmp_path / "online_maps/episode/presentation" / folder
        path.mkdir(parents=True)
        (path / name).write_bytes(b"evidence")
    monkeypatch.setattr(
        demo,
        "probe_video",
        lambda path, resolution: {
            "nb_frames": "1",
            "sha256": demo.digest(path),
            "width": resolution[0],
            "height": resolution[1],
        },
    )
    monkeypatch.setattr(
        demo, "presentation_panel_for_case", lambda *_: "observed_spatial_memory"
    )
    capture = {**demo.DEBUG_CAPTURE, "observer_views": ["presentation_ego"]}
    report = demo.verify_capture(tmp_path, case, "gpt-6.1-sol", capture=capture)
    assert set(report["media"]) == {"ego", "presentation_ego"}
    assert report["task_success"] is False
    source = demo.clean_ego_presentation(tmp_path, case, report)
    assert source["path"] == str(movie)
    assert source["layout"] == "clean_ego"
    movie.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="video changed"):
        demo.clean_ego_presentation(tmp_path, case, report)


def test_isolated_debug_plan_never_queries_hardware_or_changes_default_contract(
    monkeypatch, capsys
):
    from tarsphere.infra.simulation import demo_gpu

    def forbidden_lookup(_gpu):
        raise AssertionError("An offline plan cannot inspect GPUs.")

    monkeypatch.setattr(demo_gpu, "lookup_physical_gpu_uuid", forbidden_lookup)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "plan",
            "--debug",
            "--isolated-gpu",
            "--runtime-slot",
            "2",
            "--case",
            "navigate_and_report-tu",
        ],
    )
    assert demo.main() == 0
    capture = json.loads(capsys.readouterr().out)["contract"]["capture"]
    assert capture["navmesh_bake_backend"] == "native_gpu"
    assert capture["gpu_mapping"].startswith("single_uuid;")
    assert capture["model_observation_resolution"] == [640, 480]
    assert capture["presentation_ego_resolution"] == [640, 480]
    assert demo.CAPTURE["navmesh_bake_backend"] == "native_cpu"
    assert "gpu_mapping" not in demo.CAPTURE


def test_negative_runtime_slot_fails_before_hardware_or_api_access(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "record",
            "--gpu",
            "4",
            "--navdp-port",
            "19990",
            "--runtime-slot",
            "-1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        demo.main()
    assert error.value.code == 2
    assert "--runtime-slot must be nonnegative" in capsys.readouterr().err


@pytest.mark.parametrize("action", ["render", "verify"])
def test_debug_cannot_pass_publication_render_or_verification(action):
    result = subprocess.run(
        [demo.sys.executable, str(SCRIPT), action, "--debug"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "publication render/verify require native 2K sources" in result.stderr


def test_full_campaign_groups_recordings_by_scene():
    result = subprocess.run(
        [demo.sys.executable, str(SCRIPT), "plan"],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["contract"]["model"] == "gpt-6.1-sol"
    scene_count = len({case["scene_asset"] for case in plan["contract"]["cases"]})
    assert len(plan["commands"]) == scene_count < 12
    assert sum(command.count("--episode") for command in plan["commands"]) == 12
    assert all(
        command.count("--room-observer-view") == 2 for command in plan["commands"]
    )
    first = plan["commands"][0]
    episodes = [
        first[index + 1]
        for index, argument in enumerate(first)
        if argument == "--episode"
    ]
    assert episodes[0].endswith("/base.yaml") and episodes[1].endswith(
        "/task_update.yaml"
    )


def test_resume_selects_latest_attempt_for_each_case_in_shared_scene(tmp_path):
    for index, cases in [
        (1, ["navigate_and_report-base", "navigate_and_report-pr"]),
        (2, ["navigate_and_report-pr"]),
    ]:
        take = tmp_path / "scenes" / "bookstore_cafe" / f"take-{index:03d}"
        take.mkdir(parents=True)
        (take / "launch.json").write_text(json.dumps({"cases": cases}))
    assert [
        path.name
        for path in demo.case_takes(
            tmp_path, {"id": "navigate_and_report-base", "scene_name": "bookstore_cafe"}
        )
    ] == ["take-001"]
    assert [
        path.name
        for path in demo.case_takes(
            tmp_path, {"id": "navigate_and_report-pr", "scene_name": "bookstore_cafe"}
        )
    ] == ["take-001", "take-002"]


def test_artifact_must_belong_to_its_capture(tmp_path):
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    owned = tmp_path / "take-001"
    owned.mkdir()
    with pytest.raises(ValueError):
        demo.checked_file(str(outside), owned)


@pytest.mark.parametrize("width,height,fps", [(640, 480, "30/1"), (2560, 1440, "5/1")])
def test_low_resolution_or_low_rate_cannot_pass_as_2k(
    monkeypatch, tmp_path, width, height, fps
):
    movie = tmp_path / "video.mp4"
    movie.write_bytes(b"video")
    response = {
        "streams": [{"width": width, "height": height, "avg_frame_rate": fps}],
        "format": {"duration": "5"},
    }
    monkeypatch.setattr(
        demo.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(response), ""),
    )
    with pytest.raises(ValueError):
        demo.probe_video(movie, [2560, 1440])


def test_invalid_case_and_session_fail_before_any_recording():
    for arguments in [
        ("plan", "--case", "navigate_and_report-unknown"),
        ("plan", "--session", "../main_results"),
    ]:
        result = subprocess.run(
            [demo.sys.executable, str(SCRIPT), *arguments], capture_output=True
        )
        assert result.returncode != 0


@pytest.mark.parametrize(
    "family,condition,subtype,panel",
    [
        ("navigate_and_report", "pr", "reduced_resolution", "model_observation"),
        ("fetch_and_deliver", "pr", "depth_noise", "grounding_depth"),
        ("navigate_and_report", "base", None, "observed_spatial_memory"),
    ],
)
def test_render_selects_sensor_evidence_from_recorded_condition(
    monkeypatch, tmp_path, family, condition, subtype, panel
):
    case = {"id": f"{family}-{condition}", "episode_id": "episode"}
    result = {
        "episode_id": "episode",
        "observer_videos": {
            view: {"path": f"{view}.mp4", "frame_manifest": f"{view}.jsonl"}
            for view in ("presentation_ego", "follow")
        },
        "continuous_ego_videos": {"frame_manifest": "episode_ego_frames.jsonl"},
        "heartbeat_path": "episode.heartbeat.jsonl",
    }
    artifact = tmp_path / "physical.episode.json"
    artifact.write_text(
        json.dumps(
            {
                "episode_id": "episode",
                "evaluation": {
                    "task_profile": {
                        "capability_slice": "perceptual_robustness"
                        if condition == "pr"
                        else "base",
                        "subtype": subtype,
                    }
                },
            }
        )
    )
    result.update(
        episode_artifact_path=str(artifact),
        episode_artifact_sha256=demo.digest(artifact),
    )
    rollout = tmp_path / "physical.rollout.json"
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "tarsphere.rollout_artifact.v1",
                "episode_id": "episode",
                "episode_artifact_path": str(artifact),
                "episode_artifact_sha256": demo.digest(artifact),
            }
        )
    )
    result.update(
        rollout_artifact_path=str(rollout), rollout_artifact_sha256=demo.digest(rollout)
    )
    (tmp_path / "episode.json").write_text(json.dumps(result))
    commands = []
    monkeypatch.setattr(
        demo.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(demo, "verify_presentation", lambda *args: {})
    demo.render_capture(tmp_path, case)
    assert ("--show-model-observation" in commands[0]) is (panel == "model_observation")
    assert ("--show-grounding-depth" in commands[0]) is (panel == "grounding_depth")
    if panel == "model_observation":
        assert commands[0][commands[0].index("--model-input-dir") + 1] == str(
            tmp_path / "episode.model_output.inputs"
        )


@pytest.mark.parametrize(
    "rows,count",
    [
        ([{"physics_step": 9, "video_physics_step": 8}], 1),
        (
            [
                {"physics_step": 0, "video_physics_step": 0},
                {"physics_step": 9, "video_physics_step": 9},
            ],
            2,
        ),
        ([{"physics_step": 0, "video_physics_step": 0}], 2),
    ],
)
def test_future_frame_drift_or_truncation_fails_capture_verification(rows, count):
    with pytest.raises(ValueError, match="causal 30 fps"):
        demo.verify_frame_clock(rows, count, "side")


@pytest.mark.parametrize(
    "response_model", ["gpt-6.1-sol-2026-09-29", "different-model"]
)
def test_transport_preflight_checks_real_sdk_reply_and_keeps_credentials_private(
    monkeypatch, tmp_path, response_model
):
    requests = []
    monkeypatch.setenv("TARSPHERE_AGENT_API_KEY", "private-preflight-key")
    monkeypatch.setenv("TARSPHERE_AGENT_API_BASE", "https://example.test/v1")

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "private-preflight-key"
            assert kwargs["max_retries"] == 0 and kwargs["timeout"] == 30.0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                model=response_model,
                usage=None,
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))
                ],
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    if response_model == "different-model":
        with pytest.raises(OSError, match="preflight failed"):
            demo.check_model_transport(tmp_path, "gpt-6.1-sol")
    else:
        assert demo.check_model_transport(tmp_path, "gpt-6.1-sol")["status"] == "passed"
    text = (tmp_path / "transport-preflight.json").read_text()
    receipt = json.loads(text)
    assert "private-preflight-key" not in text and "example.test" not in text
    assert receipt["provider_model"] == response_model
    assert requests[0]["model"] == "gpt-6.1-sol"
    assert requests[0]["response_format"] == {"type": "json_object"}


def test_transport_failure_cannot_launch_and_retains_only_safe_exception_class(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TARSPHERE_AGENT_API_KEY", "private-preflight-key")
    monkeypatch.setenv("TARSPHERE_AGENT_API_BASE", "https://example.test/v1")

    def broken_client(**_kwargs):
        raise ConnectionError("Authorization: private-preflight-key")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=broken_client))
    with pytest.raises(OSError, match="ConnectionError") as error:
        demo.check_model_transport(tmp_path, "gpt-6.1-sol")
    text = (tmp_path / "transport-preflight.json").read_text()
    assert "private-preflight-key" not in text and "private-preflight-key" not in str(
        error.value
    )
    assert json.loads(text)["status"] == "failed"


def test_teleop_locomotion_needs_no_online_map_but_evaluation_navigation_does():
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv

    env = object.__new__(IsaacAutonomousEnv)
    env._private_mapping_observer = None
    env._private_mapping_action_space = ["move_body", "wait"]
    IsaacAutonomousEnv._capture_private_mapping_tick(env, "move_body", 81)
    for navigation in (
        "navigate_to_frontier",
        "navigate_to_pointgoal",
        "navigate_to_entity",
    ):
        env._private_mapping_action_space = ["move_body", navigation]
        with pytest.raises(RuntimeError, match="no private mapping observer"):
            IsaacAutonomousEnv._capture_private_mapping_tick(env, "move_body", 81)
