"""Still capture freezes native evidence and publishes complete bundles only."""

from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest

from tarsphere.infra.perception import teleop_capture as capture


@pytest.fixture
def native(monkeypatch):
    state = {
        "physics_step": 100,
        "engine_physics_step": 120,
        "simulation_time_s": 0.5,
        "timeline_time_s": 0.5,
        "timeline_playing": True,
        "actor_roots": {"robot:g1": "/World/G1", "human:operator": "/World/Person"},
        "robot": {"position": [0.0, 0.0, 0.8], "joint_positions": [0.1, 0.2]},
        "humans": {"operator": {"joints": {"Hip": [0.0, 0.1, 0.2]}}},
        "ego_camera": {"optics": {"focalLength": 18.0}},
        "viewport_camera": {"path": "/ViewportCamera"},
    }

    class Native:
        instances = []
        on_render = None
        on_close = None
        reads = []

        def __init__(self, env):
            self.closed = False
            self.layers = False
            self.instances.append(self)

        def snapshot(self):
            self.reads.append(deepcopy(state))
            return deepcopy(state)

        def prepare(self, resolution, actor_layers):
            assert resolution == (4, 3)
            self.layers = actor_layers
            return {"source_path": "/G1/ego", "optics": {"focalLength": 18.0}}

        def render(self):
            if self.on_render:
                self.on_render()
            rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            ids = np.array([[0, 2, 3, 4], [5, 2, 3, 0], [0, 0, 0, 0]], dtype=np.uint32)
            labels = {
                "0": "BACKGROUND",
                "2": "/World/G1/hand",
                "3": "/World/Person/body",
                "4": "/World/G10/hand",
                "5": "/World/G1/pelvis",
            }
            return rgb, (ids, labels) if self.layers else None

        def close(self):
            self.closed = True
            if self.on_close:
                self.on_close()

    monkeypatch.setattr(capture, "_NativeCapture", Native)
    return Native, state


def test_bundle_has_native_rgb_visible_actor_layers_and_complete_metadata(
    tmp_path, native
):
    runtime, original = native
    recorder = capture.TeleopCapture(object(), tmp_path, (4, 3), actor_layers=True)
    # Construction is lazy and does not allocate a native camera or renderer.
    assert not runtime.instances
    receipt = recorder.capture("pick_object")
    bundle = Path(receipt["directory"])
    assert bundle.is_relative_to(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [bundle.name]
    assert json.loads((bundle / "metadata.json").read_text()) == receipt
    assert receipt["state"] == original
    assert receipt["freeze_verified"]
    assert runtime.instances[0].closed
    rgb = np.asarray(Image.open(bundle / "rgb.png"))
    assert np.array_equal(rgb, np.arange(36, dtype=np.uint8).reshape(3, 4, 3))
    robot = receipt["files"]["actor_layers"]["robot:g1"]
    assert robot["visible_pixels"] == 3
    rgba = np.asarray(Image.open(bundle / robot["path"]))
    assert rgba[0, 1, 3] == 255 and rgba[1, 0, 3] == 255
    # G10 is not a descendant of G1; occluded/background pixels stay transparent.
    assert rgba[0, 3].tolist() == [0, 0, 0, 0]
    assert rgba[2, 2].tolist() == [0, 0, 0, 0]
    assert np.array_equal(rgba[0, 1, :3], rgb[0, 1])


def test_rgb_only_repeated_labels_and_close(tmp_path, native):
    runtime, _ = native
    recorder = capture.TeleopCapture(object(), tmp_path, (4, 3))
    a = recorder.capture()
    b = recorder.capture()
    assert a["directory"] != b["directory"]
    assert a["files"] == {"rgb": "rgb.png", "actor_layers": {}}
    assert all(item.closed for item in runtime.instances)
    recorder.close()
    recorder.close()
    with pytest.raises(RuntimeError, match="closed"):
        recorder.capture()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(physics_step=s["physics_step"] + 1),
        lambda s: s.update(engine_physics_step=s["engine_physics_step"] + 1),
        lambda s: s.update(timeline_time_s=s["timeline_time_s"] + 0.01),
        lambda s: s.update(timeline_playing=False),
        lambda s: s["robot"]["joint_positions"].__setitem__(1, 0.3),
        lambda s: s["humans"]["operator"]["joints"]["Hip"].__setitem__(1, 0.3),
        lambda s: s["ego_camera"]["optics"].update(focalLength=20.0),
        lambda s: s["viewport_camera"].update(path="/DifferentCamera"),
    ],
)
def test_changed_state_is_rejected_without_publishing(tmp_path, native, mutate):
    runtime, state = native
    runtime.on_render = staticmethod(lambda: mutate(state))
    recorder = capture.TeleopCapture(object(), tmp_path, (4, 3))
    with pytest.raises(RuntimeError, match="no files were published"):
        recorder.capture()
    assert list(tmp_path.iterdir()) == []
    assert runtime.instances[0].closed


def test_changes_during_native_resource_release_are_also_rejected(tmp_path, native):
    runtime, state = native
    runtime.on_close = staticmethod(lambda: state.update(physics_step=101))
    with pytest.raises(RuntimeError, match="changed physics"):
        capture.TeleopCapture(object(), tmp_path, (4, 3)).capture()
    assert not list(tmp_path.iterdir())


def test_failed_render_releases_resources_and_allows_a_later_attempt(tmp_path, native):
    runtime, _ = native

    def fail():
        raise RuntimeError("native renderer failed")

    runtime.on_render = staticmethod(fail)
    recorder = capture.TeleopCapture(object(), tmp_path, (4, 3))
    with pytest.raises(RuntimeError, match="native renderer failed"):
        recorder.capture()
    assert runtime.instances[0].closed
    assert not list(tmp_path.iterdir())
    runtime.on_render = None
    assert Path(recorder.capture()["directory"]).is_dir()


def test_encoding_failure_and_collision_never_publish_partial_files(
    tmp_path, native, monkeypatch
):
    recorder = capture.TeleopCapture(object(), tmp_path, (4, 3))
    save = Image.Image.save

    def failed_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", failed_save)
    with pytest.raises(OSError, match="disk full"):
        recorder.capture()
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(Image.Image, "save", save)
    monkeypatch.setattr(capture, "uuid4", lambda: SimpleNamespace(hex="same"))
    existing = tmp_path / "frame_same"
    existing.mkdir()
    (existing / "original.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        recorder.capture()
    assert (existing / "original.txt").read_text() == "keep"
    assert list(tmp_path.iterdir()) == [existing]


def test_invalid_paths_labels_resolution_and_unmapped_instances(tmp_path, native):
    with pytest.raises(ValueError, match="outside the source"):
        capture.TeleopCapture(
            object(), Path(capture.__file__).resolve().parents[3] / "captures"
        )
    for size in [(0, 3), (4, -1), (True, 3), (4.5, 3)]:
        with pytest.raises(ValueError, match="positive integers"):
            capture.TeleopCapture(object(), tmp_path, size)
    recorder = capture.TeleopCapture(object(), tmp_path, (4, 3))
    for label in ["../outside", ".", "", "name/other", "with spaces", "取物", "a.b"]:
        with pytest.raises(ValueError, match="label"):
            recorder.capture(label)
    with pytest.raises(ValueError, match="do not cover"):
        capture._actor_masks(np.array([[2]], dtype=np.uint32), {}, {"robot": "/G1"})
    with pytest.raises(ValueError, match="native USD path"):
        capture._actor_masks(
            np.array([[2]], dtype=np.uint32), {"2": "guessed robot"}, {"robot": "/G1"}
        )


def test_native_render_is_zero_time_and_restores_global_setting_on_error(monkeypatch):
    values = {"/app/player/playSimulations": True}
    settings = SimpleNamespace(
        get=lambda p: values.get(p),
        set_bool=lambda p, v: values.__setitem__(p, v),
        set=lambda p, v: values.__setitem__(p, v),
        destroy_item=lambda p: values.pop(p, None),
    )
    carb = ModuleType("carb")
    carb.settings = SimpleNamespace(get_settings=lambda: settings)
    omni = ModuleType("omni")
    omni.replicator = ModuleType("omni.replicator")
    calls = []

    def step(**kwargs):
        assert values["/app/player/playSimulations"] is False
        calls.append(kwargs)

    rep = ModuleType("omni.replicator.core")
    rep.orchestrator = SimpleNamespace(step=step)
    omni.replicator.core = rep
    for name, value in {
        "carb": carb,
        "carb.settings": carb.settings,
        "omni": omni,
        "omni.replicator": omni.replicator,
        "omni.replicator.core": rep,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    updates = []
    runtime = capture._NativeCapture.__new__(capture._NativeCapture)
    runtime.env = SimpleNamespace(_record_render=lambda: None)
    runtime.product = SimpleNamespace(
        hydra_texture=SimpleNamespace(set_updates_enabled=updates.append)
    )
    runtime.annotators = {
        "rgb": SimpleNamespace(get_data=lambda: np.ones((3, 4, 4), dtype=np.uint8))
    }
    rgb, segmentation = runtime.render()
    assert rgb.shape == (3, 4, 3) and segmentation is None
    assert calls == [
        dict(rt_subframes=1, delta_time=0.0, pause_timeline=False, wait_for_render=True)
    ]
    assert updates == [True, False]
    assert values["/app/player/playSimulations"] is True

    # Native failures restore the caller setting, including absence, and disarm RP.
    values.clear()
    rep.orchestrator.step = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("render error")
    )
    with pytest.raises(RuntimeError, match="render error"):
        runtime.render()
    assert not values and updates[-1] is False


def test_snapshot_reads_live_dofs_and_native_human_joints_not_cached_spawn_poses(
    monkeypatch,
):
    live = {"robot_joint": 0.2, "human_joint": 1.0}
    clock = SimpleNamespace(get_current_time=lambda: 0.5, is_playing=lambda: True)
    omni = ModuleType("omni")
    omni.timeline = SimpleNamespace(get_timeline_interface=lambda: clock)
    omni.anim = ModuleType("omni.anim")
    omni.anim.graph = ModuleType("omni.anim.graph")
    character = object()
    omni.anim.graph.core = SimpleNamespace(get_character=lambda path: character)
    simulation = ModuleType("isaacsim.core.simulation_manager")
    simulation.SimulationManager = SimpleNamespace(
        get_num_physics_steps=lambda: 100, get_simulation_time=lambda: 0.5
    )
    prim = SimpleNamespace(IsA=lambda schema: True, GetPath=lambda: "/Human/Skeleton")

    class Skeleton:
        def __init__(self, prim):
            pass

        def GetJointsAttr(self):
            return SimpleNamespace(Get=lambda: ["Hip", "Hip/Arm"])

    pxr = ModuleType("pxr")
    pxr.Usd = SimpleNamespace(PrimRange=lambda root: [prim])
    pxr.UsdGeom = SimpleNamespace()
    pxr.UsdSkel = SimpleNamespace(Skeleton=Skeleton)
    for name, value in {
        "omni": omni,
        "omni.timeline": omni.timeline,
        "omni.anim": omni.anim,
        "omni.anim.graph": omni.anim.graph,
        "omni.anim.graph.core": omni.anim.graph.core,
        "isaacsim": ModuleType("isaacsim"),
        "isaacsim.core": ModuleType("isaacsim.core"),
        "isaacsim.core.simulation_manager": simulation,
        "pxr": pxr,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(
        capture,
        "read_character_transform",
        lambda char, joint=None: {
            "position_stage_units": [live["human_joint"] if joint else 0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )
    runtime = capture._NativeCapture.__new__(capture._NativeCapture)
    runtime.stage = SimpleNamespace(GetPrimAtPath=lambda path: prim)
    runtime.source_path = "/Robot/Camera"
    runtime.camera_description = lambda path: {"path": path}
    robot = SimpleNamespace(
        get_robot_base=lambda: SimpleNamespace(
            get_pose=lambda: ([9.0, 8.0, 0.8], [1.0, 0.0, 0.0, 0.0])
        ),
        articulation=SimpleNamespace(get_joint_positions=lambda: [live["robot_joint"]]),
    )
    runtime.env = SimpleNamespace(
        _live_robots=lambda: [robot],
        _robot_root_prim_path=lambda: "/Robot",
        _human_body=SimpleNamespace(
            _character_prim_paths={"operator": "/Human"},
            _scene_bound_ids=set(),
            _character_skelroot_paths={"operator": "/Human"},
        ),
        episode=SimpleNamespace(
            robot=SimpleNamespace(entity_name="g1"),
            humans=[SimpleNamespace(id="operator")],
        ),
        # Deliberately stale caches must not mask live articulation/animation.
        _obs={"position": [0.0, 0.0, 0.0]},
        _human_positions={"operator": [0.0, 0.0, 0.0]},
        _g1_attachment=None,
        headless=True,
        physics_steps=100,
    )
    before = runtime.snapshot()
    assert before["robot"]["position"] == [9.0, 8.0, 0.8]
    assert set(before["humans"]["operator"]["joints"]["/Human/Skeleton"]) == {
        "Hip",
        "Hip/Arm",
    }
    live["robot_joint"] = 0.3
    assert not capture._same_state(before, runtime.snapshot())
    live["robot_joint"] = 0.2
    live["human_joint"] = 2.0
    assert not capture._same_state(before, runtime.snapshot())


def test_camera_copy_uses_native_render_aspect_and_serializes_vector_optics(
    monkeypatch,
):
    class Attribute:
        def __init__(self, values, name):
            self.values, self.name = values, name

        def Get(self):
            return self.values.get(self.name)

        def Set(self, value):
            self.values[self.name] = value

    class Vector:
        def __init__(self, *values):
            self.values = values

        def __iter__(self):
            return iter(self.values)

    source_values = {
        "horizontalAperture": 24.0,
        # Authored USD is stale, while the real ego RP is 640x480.
        "verticalAperture": 13.5,
        "focalLength": 18.0,
        "projection": "perspective",
        "clippingRange": Vector(0.01, 100.0),
        "clippingPlanes": [SimpleNamespace(x=1.0, y=0.0, z=0.0, w=0.0)],
    }
    prims = {}

    class Prim:
        def __init__(self, values):
            self.values = values

        def GetAttribute(self, name):
            return Attribute(self.values, name)

    class Camera:
        def __init__(self, prim):
            self.prim = prim

        @classmethod
        def Define(cls, stage, path):
            prims[path] = Prim({})
            return cls(prims[path])

        @staticmethod
        def GetSchemaAttributeNames(includeInherited=False):
            return tuple(source_values)

        def GetPrim(self):
            return self.prim

        def GetHorizontalApertureAttr(self):
            return self.prim.GetAttribute("horizontalAperture")

        def GetVerticalApertureAttr(self):
            return self.prim.GetAttribute("verticalAperture")

        def MakeMatrixXform(self):
            return self.prim.GetAttribute("local_matrix")

        def ComputeLocalToWorldTransform(self, time):
            return np.eye(4)

    pxr = ModuleType("pxr")
    pxr.UsdGeom = SimpleNamespace(Camera=Camera)
    pxr.Gf = SimpleNamespace(Matrix4d=lambda diagonal: np.eye(4) * diagonal)
    pxr.Usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None))
    omni = ModuleType("omni")
    omni.replicator = ModuleType("omni.replicator")
    rep = ModuleType("omni.replicator.core")
    product = SimpleNamespace(
        hydra_texture=SimpleNamespace(set_updates_enabled=lambda enabled: None),
        destroy=lambda: None,
    )
    created_products = []
    rep.create = SimpleNamespace(
        render_product=lambda path, resolution: (
            created_products.append((path, resolution)) or product
        )
    )
    rep.AnnotatorRegistry = SimpleNamespace(
        get_annotator=lambda name, init_params: SimpleNamespace(
            attach=lambda rp: None, detach=lambda rp: None
        )
    )
    omni.replicator.core = rep
    for name, value in {
        "pxr": pxr,
        "omni": omni,
        "omni.replicator": omni.replicator,
        "omni.replicator.core": rep,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    runtime = capture._NativeCapture.__new__(capture._NativeCapture)
    runtime.source_path = "/Robot/Camera"
    runtime.camera_path = runtime.source_path + "/Capture"
    prims[runtime.source_path] = Prim(source_values)
    runtime.stage = SimpleNamespace(
        GetPrimAtPath=lambda path: prims.get(path),
        RemovePrim=lambda path: prims.pop(path),
    )
    runtime.env = SimpleNamespace(
        _robot_camera_sensor=lambda: SimpleNamespace(resolution=(640, 480))
    )
    runtime.product = None
    runtime.annotators = {}
    metadata = runtime.prepare((2560, 1920), True)
    assert created_products == [(runtime.camera_path, (2560, 1920))]
    assert metadata["source_resolution"] == [640, 480]
    assert source_values["verticalAperture"] == 13.5
    assert metadata["optics"]["verticalAperture"] == 18.0
    assert metadata["optics"]["clippingRange"] == [0.01, 100.0]
    assert metadata["optics"]["clippingPlanes"] == [[1.0, 0.0, 0.0, 0.0]]
    json.dumps(metadata, allow_nan=False)
    assert np.array_equal(prims[runtime.camera_path].values["local_matrix"], np.eye(4))
    runtime.close()
    assert runtime.source_path in prims and runtime.camera_path not in prims
    with pytest.raises(ValueError, match="native ego aspect ratio"):
        runtime.prepare((1920, 1080), False)
    assert len(created_products) == 1
