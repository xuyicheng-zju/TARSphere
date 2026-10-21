"""The release exposes one mechanism and rejects experiment overrides."""

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tarsphere.harness.agent import project_agent_feedback
from tarsphere.harness.core.types import CameraViewObservation
from tarsphere.harness.harness import (
    AgentHarnessConfig,
    ControllerConfig,
    ProtocolConfig,
)
from tarsphere.harness.harness_cli import (
    add_agent_harness_arguments,
    agent_harness_config_from_args,
)
from tarsphere.harness.kernel.envelope_kernel import EnvelopeKernel
from tarsphere.harness.memory.bev_map import BevMap
from tarsphere.harness.tools.atomic import register_atomic_tools
from tarsphere.harness.tools.registry import ToolRegistry
from tarsphere.infra.control.internnav import (
    InternNavServerError,
    InternNavSettings,
    build_internnav_backend,
    internav_backend_available,
)
from tarsphere.infra.control.robot_body_executor import RobotBodyExecutor


def test_bev_free_space_stops_at_observed_obstacles(monkeypatch):
    monkeypatch.setenv("TARSPHERE_BEV_FREE_SPACE", "cone")
    bev = BevMap(size_m=5.0, pixels_per_meter=10, agent_radius_m=0.0)
    bev.update(np.array([[2.0, 0.0, 1.0]]), (0.0, 0.0), 0.0)
    cells = bev.world_to_cell(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    free = bev.free_mask()
    assert [bool(free[row, col]) for row, col in cells] == [True, False, False]
    assert bev.snapshot().occupied_cells == 1
    bev.reset()
    bev.update(np.empty((0, 3)), (0.0, 0.0), 0.0)
    assert not bev.free_mask().any()


def test_legacy_internnav_flags_cannot_fake_or_disable_navigation():
    legacy_fake = {"TARSPHERE_INTERNNAV_BACKEND": "fake"}
    assert not internav_backend_available(legacy_fake)
    with pytest.raises(InternNavServerError, match="requires"):
        build_internnav_backend(legacy_fake)
    connection = {"TARSPHERE_INTERNNAV_HOST": "127.0.0.1"}
    legacy_disabled = {**connection, "TARSPHERE_INTERNNAV_NAVIGATE": "off"}
    assert internav_backend_available(legacy_disabled)
    assert InternNavSettings.from_environ(
        legacy_disabled
    ) == InternNavSettings.from_environ(connection)


def test_body_execution_keeps_default_window_with_legacy_balance_flag(monkeypatch):
    monkeypatch.setenv("TARSPHERE_G1_BALANCE_INTERRUPT", "1")
    executor = RobotBodyExecutor(post_motion_settle_seconds=0.0)
    actions = []

    def step(*, action):
        actions.append(action)
        return {}, 0.0, False, False, {}

    command = {"move_by_speed": [0.1, 0.0, 0.0]}
    result = executor.step_simulation(
        env=SimpleNamespace(step=step),
        action_name="move_body",
        env_action=command,
        physics_step_budget=2,
        step_human_bodies=lambda dt: None,
        pump_kit_updates=lambda **kwargs: None,
    )
    assert result.physics_steps == 2
    assert result.balance_control == {"intervened": False}
    assert actions == [command, command]
    with pytest.raises(TypeError):
        RobotBodyExecutor(balance_interrupt_enabled=True)


def test_human_collision_setup_cannot_skip_missing_backend(monkeypatch):
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv

    monkeypatch.setenv("TARSPHERE_PROVISIONAL_HUMANS_NONBLOCKING", "1")
    monkeypatch.setitem(sys.modules, "omni.usd", None)
    env = object.__new__(IsaacAutonomousEnv)
    env._env = object()
    env._human_positions = {"person": [0.0, 0.0, 0.0]}
    env._human_collision_errors = {}
    statuses = []
    env._human_body = SimpleNamespace(
        set_physical_status=lambda person, status, **kwargs: statuses.append(status)
    )
    with pytest.raises(RuntimeError, match="Human collision backend is unavailable"):
        env._setup_human_collision_proxies()
    assert statuses == ["collision_proxy_unavailable"]


@pytest.mark.parametrize(
    "group, field, value",
    [
        ("protocol", "visual_modality", "rgbd"),
        ("protocol", "lidar_access", False),
        ("protocol", "action_residuals", False),
        ("protocol", "semantic_spatial_memory", False),
        ("protocol", "self_managed_memory", False),
        ("protocol", "grasp_skill", False),
        ("controller", "locomotion", "kinematic"),
        ("controller", "grounding", "oracle"),
        ("controller", "attachment", "fixed"),
        ("controller", "navigation_tool", "x_navdp"),
        ("controller", "human_behavior", "llm"),
    ],
)
def test_removed_mechanisms_rejected_by_config_and_cli(group, field, value):
    with pytest.raises(ValueError, match="unknown"):
        AgentHarnessConfig.from_mapping({group: {field: value}})
    constructor = ProtocolConfig if group == "protocol" else ControllerConfig
    with pytest.raises(TypeError):
        constructor(**{field: value})
    parser = argparse.ArgumentParser()
    add_agent_harness_arguments(parser)
    flag = "--" + field.replace("_", "-")
    with pytest.raises(SystemExit):
        parser.parse_args([flag] if isinstance(value, bool) else [flag, str(value)])
    if isinstance(value, bool):
        with pytest.raises(SystemExit):
            parser.parse_args(["--no-" + field.replace("_", "-")])


def test_supported_timing_and_history_parameters_round_trip():
    parser = argparse.ArgumentParser()
    add_agent_harness_arguments(parser)
    args = parser.parse_args(
        [
            "--text-history-horizon",
            "all",
            "--visual-history-horizon",
            "4",
            "--action-horizon",
            "3",
            "--camera-views",
            "ego,right_wrist",
            "--online-mapping-hz",
            "2.5",
        ]
    )
    config = agent_harness_config_from_args(args)
    assert (
        AgentHarnessConfig.from_mapping(json.loads(json.dumps(config.as_dict())))
        == config
    )
    assert config.protocol.action_horizon_limit == 3
    assert config.controller.online_mapping_hz == 2.5


def test_default_tools_include_memory_grasp_and_online_navigation():
    tools = ToolRegistry()
    register_atomic_tools(
        tools,
        semantic_backend=object(),
        grasp_backend=object(),
        memory=object(),
        navigation_backend=object(),
        navigation_fallback_backend=object(),
    )
    assert {
        "remember_entity",
        "inspect_entity",
        "grasp_object",
        "navigate_to_pointgoal",
        "navigate_to_entity",
        "navigate_to_frontier",
    } <= set(tools.spec_names())
    feedback = {"executed": True, "motion_outcome": {"realized_m": 0.9}}
    assert project_agent_feedback(feedback) == feedback
    assert project_agent_feedback(feedback) is not feedback


def test_agent_images_are_rgb_while_metric_depth_stays_private():
    kernel = EnvelopeKernel(vlm=None, tools=ToolRegistry(), protocol=ProtocolConfig())
    view = CameraViewObservation(
        name="ego",
        rgb=np.zeros((8, 12, 3), dtype=np.uint8),
        depth=np.ones((8, 12), dtype=np.float32),
    )
    memory = SimpleNamespace(
        visual_history=[SimpleNamespace(action_step=0, views={"ego": view})]
    )
    content = []
    kernel._append_visual_history(content, memory)
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert "ego RGB" in content[0]["text"]
    assert "depth" not in content[0]["text"].lower()


def test_mock_default_locomotion_reports_the_physical_controller():
    from tarsphere.infra.simulation.mock_env import MockEnv

    env = MockEnv()
    _, _, feedback = env.step({"action": "move_body", "delta": [0.2, 0.0, 0.0]})
    assert feedback["controller_mode"] == "standard"
    assert feedback["executed"] is True


def test_g1_fixed_retention_api_is_rejected():
    from tarsphere.infra.manipulation.g1_attachment import G1AttachmentExecutor

    with pytest.raises(TypeError, match="retention"):
        G1AttachmentExecutor(
            [],
            robot_root_path="/G1",
            object_local_unit_in_meters=1.0,
            retention="fixed",
        )


def test_private_certification_trajectory_is_ineligible_for_official_metrics():
    from tarsphere.evaluation.engine import require_official_trajectory
    from tarsphere.infra.episode.episode_runner import AutonomousEpisodeRunner

    class CaptureTrajectory:
        def __init__(self, **values):
            self.__dict__.update(values)

    # Exercise the runner's provenance projection without a live simulator.
    from unittest.mock import patch

    runner = object.__new__(AutonomousEpisodeRunner)
    runner.episode = SimpleNamespace(
        episode_id="private_reference",
        evaluation=SimpleNamespace(task_profile=object()),
    )
    runner.agent = SimpleNamespace(certification_only=True)
    runner._evaluation_steps = [object()]
    runner.termination_reason = "completed"
    with patch(
        "tarsphere.infra.episode.episode_runner.EpisodeTrajectory", CaptureTrajectory
    ):
        trajectory = runner.evaluation_trajectory()
    with pytest.raises(ValueError, match="certification_policy"):
        require_official_trajectory(trajectory)


def test_saved_private_certification_result_is_never_scored(tmp_path):
    from tarsphere.evaluation.online_scoring import score_saved_result

    result_path = tmp_path / "reference.json"
    result_path.write_text(json.dumps({"diagnostic_only": True}))
    result = score_saved_result(authoritative_result_path=result_path)
    assert result["success"] is False
    assert result["outcome_status"] == "evaluation_failed"
    assert "ineligible for official scoring" in result["evaluation_report"]["error"]
