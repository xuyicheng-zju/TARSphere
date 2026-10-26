"""Regressions for public G1 USDs that omit the evaluated ego camera."""

from types import SimpleNamespace
import subprocess
import sys

import pytest

from tarsphere.embodiment.g1_asset import (
    G1_CAMERA_REST_ORIENTATION_WXYZ,
    G1_CAMERA_TRANSLATION_M,
    G1_BODY_COLLISION_LINK_NAMES,
    G1_DEX3_DONOR_ROOT_PRIM,
    G1_STABLE_ROOT_PRIM,
    _g1_camera_overlay,
    _g1_donor_joint_paths,
)
from tarsphere.embodiment.dex3 import DEX3_JOINT_NAMES, DEX3_LINK_NAMES

Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")


def test_missing_camera_is_composed_at_evaluated_rest_pose(tmp_path):
    source = tmp_path / "g1.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, G1_STABLE_ROOT_PRIM).GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, f"{G1_STABLE_ROOT_PRIM}/d435_link")
    stage.GetRootLayer().Save()
    before = source.read_bytes()
    layer = Usd.Stage.CreateInMemory()
    assert layer.GetRootLayer().ImportFromString(
        f'#usda 1.0\ndef Xform "g1_29dof" (prepend references = @{source}@)\n'
        "{\n" + _g1_camera_overlay(source) + "\n}\n"
    )
    camera = layer.GetPrimAtPath(f"{G1_STABLE_ROOT_PRIM}/d435_link/Camera")
    assert camera.IsA(UsdGeom.Camera)
    assert tuple(camera.GetAttribute("xformOp:translate").Get()) == pytest.approx(
        G1_CAMERA_TRANSLATION_M
    )
    q = camera.GetAttribute("xformOp:orient").Get()
    assert (q.GetReal(), *q.GetImaginary()) == pytest.approx(
        G1_CAMERA_REST_ORIENTATION_WXYZ
    )
    assert source.read_bytes() == before
    # Rest pose only: the runtime adds the evaluation mount calibration once.
    from tarsphere.embodiment.robot_profiles import get_robot_profile

    assert get_robot_profile("G1Robot").camera_mount_pitch_deg == -15.0


def test_existing_camera_is_preserved_and_wrong_prim_type_is_rejected(tmp_path):
    source = tmp_path / "g1.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, f"{G1_STABLE_ROOT_PRIM}/d435_link")
    camera_path = f"{G1_STABLE_ROOT_PRIM}/d435_link/Camera"
    UsdGeom.Camera.Define(stage, camera_path)
    stage.GetRootLayer().Save()
    assert _g1_camera_overlay(source) == ""
    stage.GetPrimAtPath(camera_path).SetTypeName("Xform")
    stage.GetRootLayer().Save()
    with pytest.raises(RuntimeError, match="not a Camera"):
        _g1_camera_overlay(source)
    stage.RemovePrim(f"{G1_STABLE_ROOT_PRIM}/d435_link")
    stage.GetRootLayer().Save()
    with pytest.raises(RuntimeError, match="camera mount is missing"):
        _g1_camera_overlay(source)


@pytest.mark.parametrize("nested", [False, True])
def test_donor_resolves_imported_and_nvidia_joint_layouts(tmp_path, nested):
    source = tmp_path / "donor.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, G1_DEX3_DONOR_ROOT_PRIM)
    names = [
        "waist_pitch_joint",
        "waist_roll_joint",
        "left_hand_palm_joint",
        "right_hand_palm_joint",
        *DEX3_JOINT_NAMES["left"],
        *DEX3_JOINT_NAMES["right"],
    ]
    for name in names:
        parent = name.removesuffix("_joint") + "_link" if nested else "joints"
        UsdPhysics.RevoluteJoint.Define(
            stage, f"{G1_DEX3_DONOR_ROOT_PRIM}/{parent}/{name}"
        )
    for link in G1_BODY_COLLISION_LINK_NAMES:
        # NVIDIA uses a Mesh here; older imports use an Xform of colliders.
        UsdGeom.Mesh.Define(stage, f"{G1_DEX3_DONOR_ROOT_PRIM}/{link}/collisions")
    for links in DEX3_LINK_NAMES.values():
        for link in links:
            UsdGeom.Xform.Define(stage, f"{G1_DEX3_DONOR_ROOT_PRIM}/{link}")
    stage.GetRootLayer().Save()
    paths = _g1_donor_joint_paths(source)
    assert set(paths) == set(names)
    assert all(
        stage.GetPrimAtPath(path).IsA(UsdPhysics.Joint) for path in paths.values()
    )
    stable = tmp_path / "stable.usda"
    stable_stage = Usd.Stage.CreateNew(str(stable))
    root = UsdGeom.Xform.Define(stable_stage, G1_STABLE_ROOT_PRIM).GetPrim()
    stable_stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stable_stage, f"{G1_STABLE_ROOT_PRIM}/d435_link")
    for link in G1_BODY_COLLISION_LINK_NAMES:
        UsdGeom.Xform.Define(stable_stage, f"{G1_STABLE_ROOT_PRIM}/{link}")
    stable_stage.GetRootLayer().Save()
    # A fresh worker must not initialize UsdSchemaRegistry before Kit has
    # registered PhysX schemas. Run the production composition in a fresh
    # interpreter and verify that its asset inspection stays in the child.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tarsphere.embodiment.g1_asset import create_g1_with_gravity_neutral_dex3; "
            "path = create_g1_with_gravity_neutral_dex3(sys.argv[1], sys.argv[2]); "
            "assert 'pxr.Usd' not in sys.modules; print(path)",
            str(stable),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    composed = Usd.Stage.Open(result.stdout.strip())
    assert composed.GetPrimAtPath(
        f"{G1_STABLE_ROOT_PRIM}/left_knee_link/body_collisions"
    ).IsA(UsdGeom.Mesh)
    assert composed.GetPrimAtPath(f"{G1_STABLE_ROOT_PRIM}/d435_link/Camera").IsA(
        UsdGeom.Camera
    )
    for links in DEX3_LINK_NAMES.values():
        for link in links:
            assert not any(
                prim.IsA(UsdPhysics.Joint)
                for prim in Usd.PrimRange(
                    composed.GetPrimAtPath(f"{G1_STABLE_ROOT_PRIM}/{link}")
                )
            )


def test_preflight_rejects_missing_donor_and_camera(tmp_path):
    from tarsphere.cli.check_environment import _robot_runtime_asset_issues

    episode = SimpleNamespace(robot=SimpleNamespace(robot_type="G1Robot"))
    source = tmp_path / "robots/g1/g1_29dof_color.usd"
    source.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, f"{G1_STABLE_ROOT_PRIM}/d435_link")
    stage.GetRootLayer().Save()
    issues = _robot_runtime_asset_issues(tmp_path, [episode])
    assert len(issues) == 1 and "Dex3 donor USD is missing" in issues[0]
    # Other robot profiles must also validate the configured Camera, rather
    # than passing merely because a robot USD file exists.
    episode.robot.robot_type = "H1Robot"
    source = tmp_path / "robots/h1/h1.usd"
    source.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(source))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/h1").GetPrim())
    stage.GetRootLayer().Save()
    issues = _robot_runtime_asset_issues(tmp_path, [episode])
    assert len(issues) == 1 and "Required Camera prim" in issues[0]
