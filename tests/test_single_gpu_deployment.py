"""Single-server deployment preserves the default mechanism and frozen episodes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tarsphere.cli.evaluate import (
    build_parser,
    evaluation_slots,
    main,
    parse_workers_per_gpu,
    worker_command,
)

ROOT = Path(__file__).resolve().parents[1]


def test_single_gpu_has_one_worker_and_cannot_reserve_its_only_gpu():
    assert [slot.name for slot in evaluation_slots(1)] == ["gpu0_slot0"]
    assert parse_workers_per_gpu("1") == (1,)
    with pytest.raises(ValueError, match="At least one Isaac worker"):
        evaluation_slots(1, reserve_gpu0_for_services=True)
    with pytest.raises(ValueError, match="requires 1 counts"):
        evaluation_slots(1, workers_per_gpu=(1, 1))


def test_endpoint_probe_uses_real_generation_without_requiring_models(monkeypatch):
    import urllib.request
    from tarsphere.cli.evaluate import ModelEndpoint, check_agent_endpoint

    monkeypatch.setenv("PRIMARY_TEST_KEY", "test-primary")
    monkeypatch.setenv("FALLBACK_TEST_KEY", "test-fallback")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: pytest.fail(
            "The model-list API is not a generation prerequisite"
        ),
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["api_key"])
        if kwargs["api_key"] == "test-primary":
            raise ConnectionError("primary unavailable")

    monkeypatch.setattr("tarsphere.cli.evaluate._check_chat_inference", generate)
    model = ModelEndpoint(
        "agent",
        "vision-model",
        "https://example.test/v1",
        "PRIMARY_TEST_KEY",
        "FALLBACK_TEST_KEY",
    )
    check_agent_endpoint(model)
    assert calls == ["test-primary", "test-fallback"]


def test_endpoint_probe_rejects_failed_generation_without_exposing_keys(monkeypatch):
    from tarsphere.cli.evaluate import ModelEndpoint, check_agent_endpoint

    monkeypatch.setenv("PRIMARY_TEST_KEY", "private-test-key")

    def fail(**kwargs):
        raise RuntimeError("provider error contains " + kwargs["api_key"])

    monkeypatch.setattr("tarsphere.cli.evaluate._check_chat_inference", fail)
    model = ModelEndpoint(
        "agent", "vision-model", "https://example.test/v1", "PRIMARY_TEST_KEY"
    )
    with pytest.raises(RuntimeError, match="endpoint is not healthy") as error:
        check_agent_endpoint(model)
    assert "private-test-key" not in str(error.value)


def test_single_server_plan_requires_no_gpu_endpoint_or_secrets(monkeypatch, capsys):
    monkeypatch.delenv("TARSPHERE_AGENT_MODEL", raising=False)
    monkeypatch.delenv("TARSPHERE_AGENT_API_BASE", raising=False)
    monkeypatch.setattr(
        "tarsphere.cli.evaluate.detected_gpu_count",
        lambda: pytest.fail("An offline plan must not probe GPUs"),
    )
    main(build_parser().parse_args([]))
    plan = json.loads(capsys.readouterr().out)
    assert plan["selected_models"] == ["agent"]
    assert plan["episode_run_count"] == 1
    assert plan["worker_topology"]["workers_per_gpu"] == [1]
    assert plan["slots"] == ["gpu0_slot0"]


def test_episode_subset_and_unknown_episode(monkeypatch, capsys):
    from tarsphere.cli.episode_inputs import DEFAULT_EPISODES, load_episode_inputs

    episode_id = load_episode_inputs(DEFAULT_EPISODES)[0][1].episode_id
    main(build_parser().parse_args(["--episode-id", episode_id]))
    plan = json.loads(capsys.readouterr().out)
    assert plan["episode_run_count"] == 1
    assert plan["scene_load_count"] == 1
    with pytest.raises(ValueError, match="Unknown episode IDs"):
        main(build_parser().parse_args(["--episode-id", "missing"]))


def test_removed_mechanisms_are_never_forwarded_to_worker():
    command = worker_command(Path("schedule.json"))
    assert "--online-mapping-hz" in command
    assert "--camera-views" in command
    for option in (
        "--navigation-tool",
        "--grounding",
        "--visual-modality",
        "--lidar-access",
        "--grasp-skill",
    ):
        assert option not in command


def test_plan_shell_does_not_require_host_environment():
    environment = dict(os.environ, TARSPHERE_PYTHON=sys.executable)
    result = subprocess.run(
        ["bash", "scripts/evaluate.sh", "--plan"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["episode_run_count"] == 1


def test_external_index_paths_are_relative_and_duplicate_ids_fail(tmp_path):
    import shutil
    from tarsphere.cli.episode_inputs import DEFAULT_EPISODES, load_episode_inputs

    shutil.copyfile(DEFAULT_EPISODES, tmp_path / "episode.yaml")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"episodes": [{"path": "episode.yaml"}]}))
    assert load_episode_inputs(index)[0][0] == tmp_path / "episode.yaml"
    index.write_text(
        json.dumps({"episodes": [{"path": "episode.yaml"}, {"path": "episode.yaml"}]})
    )
    with pytest.raises(ValueError, match="duplicate episode paths"):
        load_episode_inputs(index)
    shutil.copyfile(DEFAULT_EPISODES, tmp_path / "duplicate.yaml")
    with pytest.raises(ValueError, match="Duplicate episode ID"):
        load_episode_inputs(tmp_path)


def test_reference_cluster_options_are_removed():
    for option in (
        "--host-count",
        "--suite",
        "--rebalance-manifest",
        "--recover-company-shards",
    ):
        with pytest.raises(SystemExit):
            build_parser().parse_args([option, "obsolete"])


def test_preflight_accepts_external_scene_and_checks_declared_maps(tmp_path):
    from types import SimpleNamespace
    from tarsphere.cli.check_environment import _pinned_scene_assets

    scene = tmp_path / "custom.usda"
    scene.write_text("#usda 1.0\n")
    scene_map = {
        field: f"custom/{field}.data"
        for field in ("rgb_bev", "projection", "occupancy")
    }
    for relative in scene_map.values():
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("resource")
    episode = SimpleNamespace(
        episode_id="custom_scene",
        scene=SimpleNamespace(asset_path=str(scene)),
        world_knowledge={"scene_map": scene_map},
    )
    assert _pinned_scene_assets(tmp_path, [episode]) == (1, [])
    (tmp_path / scene_map["rgb_bev"]).unlink()
    assert _pinned_scene_assets(tmp_path, [episode]) == (1, [])
    assert (
        str(tmp_path / scene_map["rgb_bev"])
        in _pinned_scene_assets(tmp_path, [episode], require_visual_maps=True)[1]
    )
    (tmp_path / scene_map["occupancy"]).unlink()
    count, missing = _pinned_scene_assets(tmp_path, [episode])
    assert count == 1
    assert missing == [str(tmp_path / scene_map["occupancy"])]


def test_preflight_requires_only_selected_robot_assets(tmp_path):
    from types import SimpleNamespace
    from tarsphere.cli.check_environment import _selected_runtime_assets

    episode = SimpleNamespace(robot=SimpleNamespace(robot_type="G1Robot"), humans=[])
    assert _selected_runtime_assets(tmp_path, [episode]) == [
        tmp_path / "robots/g1/g1_29dof_color.usd",
        tmp_path / "robots/g1_with_hand/g1_29dof_with_hand_rev_1_0.usd",
    ]
    episode.robot.robot_type = "MobileFrankaRobot"
    assert _selected_runtime_assets(tmp_path, [episode]) == [
        tmp_path / "robots/mobile_franka/mobile_franka.usda"
    ]


def test_gpu_visibility_preserves_physical_indices_and_uuid_prefixes():
    from tarsphere.cli.gpu_selection import select_gpu_indices

    devices = [(0, "GPU-aaa"), (1, "GPU-bbb"), (2, "GPU-ccc")]
    assert select_gpu_indices(devices, "2,0") == [2, 0]
    assert select_gpu_indices(devices, "GPU-bb") == [1]
    assert select_gpu_indices(devices, "") == []
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        select_gpu_indices(devices, "2,2")


def test_isaac_priming_returns_live_environment_and_first_observation():
    # Execute this small orchestration boundary without importing Isaac or CUDA.
    import ast
    from types import SimpleNamespace

    tree = ast.parse((ROOT / "tarsphere/cli/worker.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prime_isaac_context"
    )
    function.returns = None
    for argument in function.args.args + function.args.kwonlyargs:
        argument.annotation = None
    function.args.kw_defaults = [
        None if value is None else ast.Constant(None)
        for value in function.args.kw_defaults
    ]
    observation = {"rgb": "real_first_frame"}
    environment = SimpleNamespace(reset=lambda **kwargs: observation)
    namespace = {"_create_batch_environment": lambda *args, **kwargs: environment}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "worker_priming",
            "exec",
        ),
        namespace,
    )
    episode = SimpleNamespace(scene=SimpleNamespace(asset_path="custom.usda"))
    assert namespace["_prime_isaac_context"](
        [episode], headless=True, image_dir="unused", viewer_mode="first_person"
    ) == (environment, observation)


def test_completion_budget_is_positive_and_bound_to_plan_and_schedule(capsys, tmp_path):
    from tarsphere.cli.evaluate import (
        ModelEndpoint,
        build_jobs,
        load_episodes,
        schedule_document,
    )
    from tarsphere.cli.episode_inputs import DEFAULT_EPISODES
    from tarsphere.configs.secrets import DEFAULT_AGENT_MAX_TOKENS

    assert build_parser().parse_args([]).max_tokens == DEFAULT_AGENT_MAX_TOKENS == 16384
    args = build_parser().parse_args(["--max-tokens", "2048"])
    main(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["selected_model_bindings"][0]["max_tokens"] == 2048
    model = ModelEndpoint(
        "agent", "gpt-6.1-sol", "https://example.test/v1", "KEY", max_tokens=2048
    )
    job = build_jobs([model], load_episodes(DEFAULT_EPISODES))[0]
    schedule = schedule_document(tmp_path, job)
    assert schedule["scene_runs"][0]["model_runs"][0]["max_tokens"] == 2048
    for invalid in ("0", "-1"):
        with pytest.raises(ValueError, match="--max-tokens must be positive"):
            main(build_parser().parse_args(["--max-tokens", invalid]))


def test_worker_forwards_completion_budget_to_agent_factory():
    import ast

    tree = ast.parse((ROOT / "tarsphere/cli/worker.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_worker_agent"
    )
    function.returns = None
    for argument in function.args.args + function.args.kwonlyargs:
        argument.annotation = None
    function.args.kw_defaults = [
        None if value is None else ast.Constant(None)
        for value in function.args.kw_defaults
    ]
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return object()

    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "worker_agent_boundary",
            "exec",
        ),
        {"build_robot_agent_envelope": build},
        namespace := {},
    )
    namespace["_build_worker_agent"](
        api_key="test",
        model="gpt-6.1-sol",
        api_base="https://example.test/v1",
        max_tokens=2048,
    )
    assert captured["max_tokens"] == 2048


@pytest.mark.parametrize(
    "removed",
    [
        ["--agent-mode", "model"],
        ["--agent-mode", "golden-oracle"],
        ["--agent-mode", "harness-reference"],
        ["--physical-task-only"],
    ],
)
def test_worker_parser_rejects_removed_certification_options(removed):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tarsphere/cli/worker.py"), *removed],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_resuming_cannot_relabel_saved_model_or_completion_budget():
    from copy import deepcopy
    from tarsphere.cli.evaluate import validate_plan_reuse

    saved = {
        "host_index": 0,
        "host_count": 1,
        "selected_model_bindings": [{"model": "gpt-6.1-sol", "max_tokens": 2048}],
    }
    validate_plan_reuse(saved, deepcopy(saved))
    for key, value in (("max_tokens", 16384), ("model", "another-model")):
        changed = deepcopy(saved)
        changed["selected_model_bindings"][0][key] = value
        with pytest.raises(
            ValueError, match="selected_model_bindings.*new --output-dir"
        ):
            validate_plan_reuse(saved, changed)
