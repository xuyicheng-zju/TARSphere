import pytest
import torch
import numpy as np
import trimesh
from pathlib import Path

# Trigger graspgenx's auto-setup hook (clones gripper_descriptions if needed
# and registers it on sys.path). See README → "Setup Checkpoints and Gripper
# Assets".
import graspgenx  # noqa: F401
from graspgenx import get_gripper_descriptions_assets

from graspgenx.robot import (
    GripperInfo,
    get_canonical_gripper_control_points,
    generate_circle_points,
    load_visualize_control_points_multi_suction,
    parse_offset_transform_from_yaml,
    load_gripper_yaml_file,
    load_control_points_core,
    import_module_from_path,
)
from graspgenx.x_grippers import XGripperInfo, resolve_gripper_info


def test_canonical_gripper_control_points():
    w = 0.1
    d = 0.05

    control_points = get_canonical_gripper_control_points(w, d)

    assert control_points.shape == (4, 3)

    expected_points = np.array(
        [
            [w / 2, 0, d / 2],
            [-w / 2, 0, d / 2],
            [w / 2, 0, d],
            [-w / 2, 0, d],
        ]
    )
    assert np.allclose(control_points, expected_points)


@pytest.mark.parametrize("width,depth", [(0.1, 0.05), (0.2, 0.1), (0.05, 0.02)])
def test_canonical_gripper_control_points_parameterized(width, depth):
    control_points = get_canonical_gripper_control_points(width, depth)

    assert control_points.shape == (4, 3)

    expected_points = np.array(
        [
            [width / 2, 0, depth / 2],
            [-width / 2, 0, depth / 2],
            [width / 2, 0, depth],
            [-width / 2, 0, depth],
        ]
    )
    assert np.allclose(control_points, expected_points)


def test_generate_circle_points_shape():
    center = [0.0, 0.0]
    points = generate_circle_points(center, radius=0.01, N=8)
    assert points.shape == (8, 2)


def test_generate_circle_points_radius():
    center = [0.0, 0.0]
    radius = 0.01
    points = generate_circle_points(center, radius=radius, N=30)

    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    assert np.allclose(distances, radius, atol=1e-10)


def test_generate_circle_points_center_offset():
    center = [5.0, 3.0]
    radius = 0.02
    N = 20
    points = generate_circle_points(center, radius=radius, N=N)

    assert points.shape == (N, 2)
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
    assert np.allclose(distances, radius, atol=1e-10)


def test_generate_circle_points_default_params():
    center = [0.0, 0.0]
    points = generate_circle_points(center)
    assert points.shape == (30, 2)

    distances = np.sqrt(np.sum(points ** 2, axis=1))
    assert np.allclose(distances, 0.007, atol=1e-10)


def test_load_visualize_control_points_multi_suction():
    suction_centers = [[0.0, 0.0, 0.1], [0.1, 0.1, 0.1], [-0.1, -0.1, 0.1]]

    points = load_visualize_control_points_multi_suction(suction_centers)

    assert points.shape[0] == len(suction_centers)
    assert points.shape[2] == 3

    # Z-coordinates should all match the first center's z
    z_coords = points[:, :, 2]
    assert np.allclose(z_coords, 0.1)

    # Points form circles around centers
    for i, center in enumerate(suction_centers):
        cup_points = points[i]
        distances = np.sqrt(np.sum((cup_points[:, :2] - center[:2]) ** 2, axis=1))
        assert np.allclose(distances, 0.005, atol=1e-6)


def test_parse_offset_transform_from_yaml():
    yaml_transform = [
        [0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0, 1.0],  # identity quaternion (xyzw)
    ]

    transform = parse_offset_transform_from_yaml(yaml_transform)

    assert transform.shape == (4, 4)

    expected_transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(transform, expected_transform)


def test_parse_offset_transform_invalid_input():
    with pytest.raises(AssertionError):
        parse_offset_transform_from_yaml([[0.1, 0.2]])

    with pytest.raises(AssertionError):
        parse_offset_transform_from_yaml([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])


def test_gripper_info_dataclass():
    collision_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    visual_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    offset_transform = np.eye(4)
    control_points = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    gripper_info = GripperInfo(
        gripper_name="test_gripper",
        collision_mesh=collision_mesh,
        visual_mesh=visual_mesh,
        offset_transform=offset_transform,
        control_points=control_points,
        depth=0.1,
        symmetric=True,
    )

    assert gripper_info.gripper_name == "test_gripper"
    assert isinstance(gripper_info.collision_mesh, trimesh.base.Trimesh)
    assert isinstance(gripper_info.visual_mesh, trimesh.base.Trimesh)
    assert np.allclose(gripper_info.offset_transform, offset_transform)
    assert np.allclose(gripper_info.control_points, control_points)
    assert gripper_info.depth == 0.1
    assert gripper_info.symmetric is True


def test_load_control_points_core_from_width_depth():
    config = {"width": 0.1, "depth": 0.05}
    control_points = load_control_points_core(config)
    assert control_points.shape == (4, 3)


def test_load_control_points_core_from_explicit():
    explicit_points = [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
    config = {"control_points": explicit_points}
    control_points = load_control_points_core(config)
    assert len(control_points) == 2


def test_load_control_points_core_missing_raises():
    with pytest.raises(NotImplementedError):
        load_control_points_core({})


def test_load_gripper_yaml_file(tmp_path):
    yaml_content = """
gripper_name: test_gripper
width: 0.1
depth: 0.05
symmetric: true
offset_transform:
    - [1.0, 0.0, 0.0, 0.1]
    - [0.0, 1.0, 0.0, 0.2]
    - [0.0, 0.0, 1.0, 0.3]
    - [0.0, 0.0, 0.0, 1.0]
"""
    yaml_file = tmp_path / "test_gripper.yaml"
    yaml_file.write_text(yaml_content)

    config = load_gripper_yaml_file(yaml_file)

    assert config["gripper_name"] == "test_gripper"
    assert config["width"] == 0.1
    assert config["depth"] == 0.05
    assert config["symmetric"] is True
    assert len(config["offset_transform"]) == 4


def test_gripper_descriptions_assets_root_exists():
    """The auto-cloned (or $GRASPGENX_GRIPPER_CFG_DIR) gripper_descriptions
    checkout must expose an ``assets/x_grippers/`` directory."""
    assets_root = get_gripper_descriptions_assets()
    assert assets_root.is_dir(), f"x_grippers root missing: {assets_root}"
    assert (assets_root / "franka_panda").is_dir(), (
        f"franka_panda not found under {assets_root}"
    )


def test_franka_panda_config_present():
    """franka_panda must ship the config.json + URDF that resolve_gripper_info
    consumes."""
    gripper_dir = get_gripper_descriptions_assets() / "franka_panda"
    assert (gripper_dir / "config.json").is_file()
    assert (gripper_dir / "gripper.urdf").is_file()


def test_resolve_gripper_info_franka_panda():
    gripper_info = resolve_gripper_info("franka_panda")

    assert isinstance(gripper_info, XGripperInfo)
    assert gripper_info.gripper_name == "franka_panda"
    assert isinstance(gripper_info.collision_mesh, trimesh.base.Trimesh)
    assert isinstance(gripper_info.visual_mesh, trimesh.base.Trimesh)


def test_resolve_gripper_info_franka_panda_depth():
    gripper_info = resolve_gripper_info("franka_panda")
    assert isinstance(gripper_info.depth, float)
    assert gripper_info.depth > 0


def test_resolve_gripper_info_invalid_raises():
    with pytest.raises(ValueError, match="Unknown gripper"):
        resolve_gripper_info("nonexistent_gripper")


def test_import_module_from_path(tmp_path):
    module_file = tmp_path / "test_module.py"
    module_file.write_text("ANSWER = 42\n")

    module = import_module_from_path(module_file)
    assert module.ANSWER == 42


def test_import_module_from_path_invalid():
    with pytest.raises((ImportError, FileNotFoundError, OSError)):
        import_module_from_path("/nonexistent/path/module.py")


@pytest.mark.parametrize("gripper_name", ["franka_panda", "robotiq_2f_140"])
def test_gripper_info_consistency(gripper_name):
    gripper_info = resolve_gripper_info(gripper_name)

    assert hasattr(gripper_info, "gripper_name")
    assert hasattr(gripper_info, "collision_mesh")
    assert hasattr(gripper_info, "visual_mesh")
    assert hasattr(gripper_info, "control_points")
    assert hasattr(gripper_info, "tool_tcp_transform")
    assert hasattr(gripper_info, "sweep_volume")
    assert hasattr(gripper_info, "grasp_volume")

    assert gripper_info.tool_tcp_transform.shape == (4, 4)
    assert isinstance(gripper_info.collision_mesh, trimesh.base.Trimesh)
    assert isinstance(gripper_info.visual_mesh, trimesh.base.Trimesh)
