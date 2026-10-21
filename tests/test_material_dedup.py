"""USD-only material optimization preserves shading and physical semantics."""

import hashlib
from pathlib import Path

import pytest

pxr = pytest.importorskip(
    "pxr", reason="USD-only tests require usd-core or Isaac's USD"
)
from pxr import Sdf, Usd, UsdGeom, UsdShade

from tarsphere.infra.simulation.material_dedup import (
    bake_material_overlay,
    _file_digest,
)


def material(stage, name, *, roughness=0.4, terminal="Shader", texture=None):
    result = UsdShade.Material.Define(stage, f"/World/{name}")
    for shader_name in ("Shader", "Alternate"):
        shader = UsdShade.Shader.Define(stage, f"/World/{name}/{shader_name}")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        if texture is not None:
            shader.CreateInput("texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(texture)
            )
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    result.CreateSurfaceOutput().ConnectToSource(
        UsdShade.Shader(stage.GetPrimAtPath(f"/World/{name}/{terminal}")).GetOutput(
            "surface"
        )
    )
    return result


def scene(tmp_path):
    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetTimeCodesPerSecond(240)
    stage.SetFramesPerSecond(30)
    return stage, source, tmp_path / "scene.dedup.usd"


def test_equivalent_materials_retarget_all_binding_purposes_and_preserve_source(
    tmp_path,
):
    stage, source, output = scene(tmp_path)
    material(stage, "A")
    material(stage, "B")
    mesh = UsdGeom.Cube.Define(stage, "/World/Cube").GetPrim()
    for name in (
        "material:binding",
        "material:binding:preview",
        "material:binding:physics",
    ):
        mesh.CreateRelationship(name).SetTargets(["/World/B"])
    collection = Usd.CollectionAPI.Apply(mesh, "parts")
    collection.GetIncludesRel().SetTargets(["/World/Cube"])
    mesh.CreateRelationship("material:binding:collection:parts").SetTargets(
        [collection.GetCollectionPath(), "/World/B"]
    )
    stage.GetRootLayer().Save()
    before = source.read_bytes()
    report = bake_material_overlay(source, output)
    assert report["deactivated"] == 1
    assert report["rewritten_bindings"] == 4
    assert source.read_bytes() == before
    composed = Usd.Stage.Open(str(output))
    assert composed.GetPrimAtPath("/World/A").IsActive()
    assert not composed.GetPrimAtPath("/World/B").IsActive()
    assert composed.GetPrimAtPath("/World/Cube").GetRelationship(
        "material:binding:physics"
    ).GetTargets() == [Sdf.Path("/World/A")]
    assert composed.GetMetadata("upAxis") == "Z"
    assert composed.GetMetadata("metersPerUnit") == 0.01
    assert composed.GetTimeCodesPerSecond() == 240
    assert composed.GetFramesPerSecond() == 30
    assert composed.GetDefaultPrim().GetPath() == Sdf.Path("/World")


@pytest.mark.parametrize(
    "difference", ["value", "terminal", "connection", "physics", "animated", "api"]
)
def test_unequal_networks_and_physics_materials_are_not_merged(tmp_path, difference):
    stage, source, output = scene(tmp_path)
    material(stage, "A")
    second = material(
        stage,
        "B",
        roughness=0.8 if difference == "value" else 0.4,
        terminal="Alternate" if difference == "terminal" else "Shader",
    )
    if difference == "connection":
        for name in ("A", "B"):
            attribute = stage.GetPrimAtPath(f"/World/{name}/Shader").CreateAttribute(
                "inputs:linked", Sdf.ValueTypeNames.Token
            )
            node = "Shader" if name == "A" else "Alternate"
            attribute.SetConnections([f"/World/{name}/{node}.outputs:surface"])
    elif difference == "physics":
        second.GetPrim().CreateAttribute(
            "physics:staticFriction", Sdf.ValueTypeNames.Float
        ).Set(0.9)
    elif difference == "animated":
        stage.GetPrimAtPath("/World/B/Shader").GetAttribute("inputs:roughness").Set(
            0.7, 1
        )
    elif difference == "api":
        second.GetPrim().SetMetadata(
            "apiSchemas", Sdf.TokenListOp.CreateExplicit(["PhysicsMaterialAPI"])
        )
    stage.GetRootLayer().Save()
    report = bake_material_overlay(source, output)
    assert report["deactivated"] == 0
    assert output.is_file()  # A no-op bake is still a valid cached overlay.


def test_nonbinding_shader_reference_keeps_duplicate_alive(tmp_path):
    stage, source, output = scene(tmp_path)
    material(stage, "A")
    material(stage, "B")
    user = UsdShade.Shader.Define(stage, "/World/User")
    user.CreateInput("linked", Sdf.ValueTypeNames.Token).GetAttr().SetConnections(
        ["/World/B/Shader.outputs:surface"]
    )
    stage.GetRootLayer().Save()
    assert bake_material_overlay(source, output)["deactivated"] == 0


def test_assets_use_full_content_and_unresolved_assets_stay_distinct(tmp_path):
    stage, source, output = scene(tmp_path)
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "image.bin").write_bytes(b"A" * (2 << 20))
    changed = tmp_path / "two/image.bin"
    with changed.open("r+b") as stream:
        stream.seek(1 << 20)
        stream.write(b"B")
    material(stage, "A", texture="one/image.bin")
    material(stage, "B", texture="two/image.bin")
    material(stage, "MissingA", texture="missing.bin")
    material(stage, "MissingB", texture="missing.bin")
    stage.GetRootLayer().Save()
    assert bake_material_overlay(source, output)["deactivated"] == 0
    cache = {}
    asset = tmp_path / "one/image.bin"
    assert _file_digest(asset, cache) == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert len(cache) == 1


def test_equal_relative_assets_resolve_from_the_source_layer(tmp_path):
    stage, source, output = scene(tmp_path)
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "image.bin").write_bytes(b"same bytes")
    material(stage, "A", texture="one/image.bin")
    material(stage, "B", texture="two/image.bin")
    stage.GetRootLayer().Save()
    assert bake_material_overlay(source, output)["deactivated"] == 1


def test_failed_verification_preserves_previous_overlay(tmp_path, monkeypatch):
    from tarsphere.infra.simulation import material_dedup

    stage, source, output = scene(tmp_path)
    stage.GetRootLayer().Save()
    output.write_bytes(b"previous valid artifact")

    def reject(*args):
        raise RuntimeError("verification refused")

    monkeypatch.setattr(material_dedup, "_verify", reject)
    with pytest.raises(RuntimeError, match="verification refused"):
        bake_material_overlay(source, output)
    assert output.read_bytes() == b"previous valid artifact"
    assert not list(tmp_path.glob(".scene.dedup-*"))


def test_cache_rejects_old_generator_and_changed_asset_dependencies(tmp_path):
    from tarsphere.infra.simulation.material_dedup import overlay_is_current

    stage, source, output = scene(tmp_path)
    asset = tmp_path / "texture.bin"
    asset.write_bytes(b"first texture")
    material(stage, "A", texture="texture.bin")
    material(stage, "B", texture="texture.bin")
    stage.GetRootLayer().Save()
    output.write_bytes(b"old unversioned overlay")
    assert not overlay_is_current(source, output)
    bake_material_overlay(source, output)
    assert overlay_is_current(source, output)
    asset.write_bytes(b"changed texture content")
    assert not overlay_is_current(source, output)


@pytest.mark.parametrize(
    "resource", ["texture_2d(filename)", 'texture_2d("image.custom")']
)
def test_relative_mdl_resource_context_is_not_assumed_equivalent(tmp_path, resource):
    stage, source, output = scene(tmp_path)
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "material.mdl").write_text(
            f"export material sample() = {resource};\n"
        )
        (folder / "image.custom").write_bytes(name.encode())
    material(stage, "A", texture="one/material.mdl")
    material(stage, "B", texture="two/material.mdl")
    stage.GetRootLayer().Save()
    assert bake_material_overlay(source, output)["deactivated"] == 0


def test_instance_proxy_materials_remain_untouched(tmp_path):
    stage, source, output = scene(tmp_path)
    material(stage, "A")
    library = tmp_path / "asset.usda"
    referenced = Usd.Stage.CreateNew(str(library))
    UsdGeom.Xform.Define(referenced, "/World")
    material(referenced, "A")
    referenced.SetDefaultPrim(referenced.GetPrimAtPath("/World"))
    referenced.GetRootLayer().Save()
    for name in ("InstanceOne", "InstanceTwo"):
        prim = stage.DefinePrim("/World/" + name)
        prim.GetReferences().AddReference(str(library))
        prim.SetInstanceable(True)
    stage.GetRootLayer().Save()
    assert bake_material_overlay(source, output)["deactivated"] == 0
    composed = Usd.Stage.Open(str(output))
    assert composed.GetPrimAtPath("/World/InstanceOne/A").IsInstanceProxy()


def test_runtime_auto_bakes_through_owned_module_and_then_reuses_cache(
    tmp_path, monkeypatch
):
    from tarsphere.infra.simulation import scene_materials

    stage, source, _ = scene(tmp_path)
    material(stage, "A")
    material(stage, "B")
    stage.GetRootLayer().Save()
    monkeypatch.delenv("TARSPHERE_SCENE_MATERIAL_DEDUP", raising=False)
    output = scene_materials.ensure_material_deduped_usd(source)
    assert output != source and output.is_file()
    assert not Usd.Stage.Open(str(output)).GetPrimAtPath("/World/B").IsActive()

    def forbidden(*args, **kwargs):
        raise AssertionError("An unchanged verified overlay must not bake again")

    monkeypatch.setattr(scene_materials.subprocess, "run", forbidden)
    assert scene_materials.ensure_material_deduped_usd(source) == output
