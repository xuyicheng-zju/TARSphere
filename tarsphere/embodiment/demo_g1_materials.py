"""Prepare an opt-in demo donor without the 16 dangling Dex3 visual bindings.

This CPU-only preparation never changes the default robot composition or source
assets. Its returned robot is usable only after effective materials and composed
geometry/physics have compared equal. It does not certify native render speed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from tarsphere.embodiment.dex3 import DEX3_LINK_NAMES
from tarsphere.embodiment.g1_asset import (
    G1_DEX3_DONOR_ROOT_PRIM,
    G1_LOOK_DEX3_PRIM,
    G1_STABLE_ROOT_PRIM,
    _composed_g1_contents,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependency_layers(donor: Path):
    from pxr import Sdf

    layers = {}
    pending = [donor]
    while pending:
        path = pending.pop().resolve(strict=True)
        path.relative_to(donor.parent)
        if path in layers:
            continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise ValueError(f"Cannot read donor layer: {path}")
        layers[path] = layer
        for reference in layer.GetCompositionAssetDependencies():
            if reference:  # Empty asset path denotes an internal reference.
                pending.append(path.parent / reference)
    return layers


def _value(value):
    """Lossless numeric payload for CPU equality, including complete Vt arrays."""
    import numpy as np
    from pxr import Sdf

    if isinstance(value, Sdf.AssetPath):
        return {"asset": value.path, "resolved": value.resolvedPath}
    if isinstance(value, Sdf.Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in sorted(value.items())}
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return [float(value.GetReal()), *map(float, value.GetImaginary())]
    array = np.asarray(value)
    if array.dtype.kind in "biufc":
        return {
            "dtype": str(array.dtype),
            "shape": array.shape,
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if array.dtype.kind in "US":
        return array.tolist()
    raise TypeError(f"Unsupported USD value in equality audit: {type(value).__name__}")


def _snapshot(stage, removed_paths: set[str]):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    groups = {
        name: (hashlib.sha256(), 0)
        for name in (
            "all_properties",
            "geometry",
            "joints_and_physics",
            "shader_and_texture",
            "valid_material_bindings",
            "prim_structure_and_world_transforms",
        )
    }
    materials = {}

    def record(group, path, payload):
        digest, count = groups[group]
        digest.update(json.dumps([path, payload], sort_keys=True).encode() + b"\n")
        groups[group] = (digest, count + 1)

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.GetPrimIndex().localErrors:
            raise ValueError(f"USD composition errors at {prim.GetPath()}")
        path = str(prim.GetPath())
        structure = [
            str(prim.GetTypeName()),
            prim.IsActive(),
            prim.IsInstanceable(),
            list(prim.GetAppliedSchemas()),
        ]
        if prim.IsA(UsdGeom.Xformable):
            structure.append(
                _value(
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                )
            )
        record("prim_structure_and_world_transforms", path, structure)
        for attr in prim.GetAttributes():
            samples = [None, *attr.GetTimeSamples()]
            payload = [
                str(attr.GetTypeName()),
                str(attr.GetVariability()),
                attr.IsCustom(),
                [str(x) for x in attr.GetConnections()],
                [
                    [
                        time,
                        _value(
                            attr.Get(Usd.TimeCode.Default() if time is None else time)
                        ),
                    ]
                    for time in samples
                ],
            ]
            key = str(attr.GetPath())
            record("all_properties", key, payload)
            if prim.IsA(UsdGeom.Mesh):
                record("geometry", key, payload)
            if prim.IsA(UsdPhysics.Joint) or attr.GetName().startswith(
                ("physics:", "physx")
            ):
                record("joints_and_physics", key, payload)
            if prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.Material):
                record("shader_and_texture", key, payload)
        for relationship in prim.GetRelationships():
            key = str(relationship.GetPath())
            if key in removed_paths:
                continue
            targets = [str(x) for x in relationship.GetTargets()]
            payload = [
                targets,
                relationship.IsCustom(),
                relationship.GetMetadata("bindMaterialAs"),
            ]
            record("all_properties", key, payload)
            if prim.IsA(UsdPhysics.Joint) or relationship.GetName().startswith(
                ("physics:", "material:binding:physics")
            ):
                record("joints_and_physics", key, payload)
            if relationship.GetName().startswith("material:binding") and targets:
                record("valid_material_bindings", key, payload)
            if prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.Material):
                record("shader_and_texture", key, payload)
        if prim.IsA(UsdGeom.Mesh):
            material, binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            materials[path] = {
                "material": str(material.GetPath()) if material else None,
                "binding": str(binding.GetPath()) if binding else None,
            }
    return (
        {
            name: {"sha256": value.hexdigest(), "records": count}
            for name, (value, count) in groups.items()
        },
        materials,
    )


def prepare_demo_g1_materials(stable_g1: Path, donor: Path, output_dir: Path) -> dict:
    """Write a new private donor/robot and an equality receipt; never overwrite.

    Call explicitly before attaching the resulting ``prepared_robot`` to Isaac.
    Merely enabling presentation cameras does not call this preparation.
    """
    from pxr import Sdf, Usd, UsdShade

    stable_g1 = Path(stable_g1).resolve(strict=True)
    donor = Path(donor).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if any(output_dir.is_relative_to(p.parent) for p in (stable_g1, donor)):
        raise ValueError(
            "Demo preparation must be outside the source asset directories."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "tarsphere.demo.g1_material_preparation.v1",
        "status": "preparing",
        "scope": "CPU USD composition only; no native performance or image acceptance",
        "default_enabled": False,
        "usd_version": list(Usd.GetVersion()),
    }
    try:
        layers = _dependency_layers(donor)
        inputs = {str(path): _sha(path) for path in [stable_g1, *sorted(layers)]}
        receipt["source_sha256_before"] = inputs
        links = [name for names in DEX3_LINK_NAMES.values() for name in names]
        expected_target = Sdf.Path(f"{G1_DEX3_DONOR_ROOT_PRIM}/Looks/material_white")
        source_paths = {
            name: f"/visuals/{name}/{name}.material:binding" for name in links
        }
        composed_paths = {
            f"{G1_STABLE_ROOT_PRIM}/{name}/visuals/{name}.material:binding"
            for name in links
        }
        source_layers = {}
        for name, path in source_paths.items():
            matches = [
                (filename, layer.GetObjectAtPath(path))
                for filename, layer in layers.items()
                if layer.GetObjectAtPath(path) is not None
            ]
            if len(matches) != 1 or not isinstance(matches[0][1], Sdf.RelationshipSpec):
                raise ValueError(f"Expected exactly one authored relationship: {path}")
            filename, relationship = matches[0]
            if list(relationship.targetPathList.GetAppliedItems()) != [expected_target]:
                raise ValueError(f"Unexpected material target at {path}")
            source_layers[name] = filename
        baseline_path = output_dir / "baseline_robot.usda"
        baseline_path.write_text(
            _composed_g1_contents(stable_g1, donor, unlock_waist_roll=False)
        )
        before = Usd.Stage.Open(str(baseline_path))
        # Refuse removal unless the actual composed ancestor is the intended
        # valid, stronger binding, including the effective descendant meshes.
        for name in links:
            prim = before.GetPrimAtPath(f"{G1_STABLE_ROOT_PRIM}/{name}")
            material, binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if (
                not material
                or str(material.GetPath()) != G1_LOOK_DEX3_PRIM
                or not binding
                or binding.GetMetadata("bindMaterialAs") != "strongerThanDescendants"
                or str(binding.GetPath()) != str(prim.GetPath()) + ".material:binding"
            ):
                raise ValueError(f"Missing effective stronger Dex3 binding: {name}")
        before_hashes, before_materials = _snapshot(before, composed_paths)
        for name in links:
            meshes = {
                p: v
                for p, v in before_materials.items()
                if p.startswith(f"{G1_STABLE_ROOT_PRIM}/{name}/")
            }
            if not meshes or any(
                v["material"] != G1_LOOK_DEX3_PRIM for v in meshes.values()
            ):
                raise ValueError(f"Unexpected effective hand mesh material: {name}")

        private = output_dir / "donor"
        for path in layers:
            destination = private / path.relative_to(donor.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        changed = {}
        for name in links:
            source = source_layers[name]
            destination = private / source.relative_to(donor.parent)
            layer = changed.setdefault(source, Sdf.Layer.FindOrOpen(str(destination)))
            relationship = layer.GetObjectAtPath(source_paths[name])
            relationship.owner.RemoveProperty(relationship)
            if layer.GetObjectAtPath(source_paths[name]) is not None:
                raise ValueError(
                    f"Private relationship was not removed: {source_paths[name]}"
                )
        receipt["private_layer_semantic_verification"] = {}
        for source, layer in changed.items():
            layer.Save()
            # Restore only the explicit removal allowlist in an anonymous
            # layer. Exact USDA equality proves no other source field changed.
            restored = Sdf.Layer.CreateAnonymous()
            restored.TransferContent(layer)
            for name in links:
                if source_layers[name] == source:
                    Sdf.CopySpec(
                        layers[source], source_paths[name], restored, source_paths[name]
                    )
            original_text = layers[source].ExportToString()
            restored_text = restored.ExportToString()
            if original_text != restored_text:
                raise ValueError(
                    "Private donor changed fields outside the 16 binding allowlist."
                )
            receipt["private_layer_semantic_verification"][str(source)] = {
                "source_usda_sha256": hashlib.sha256(
                    original_text.encode()
                ).hexdigest(),
                "restored_private_usda_sha256": hashlib.sha256(
                    restored_text.encode()
                ).hexdigest(),
            }
        prepared = output_dir / "prepared_robot.usda"
        prepared.write_text(
            _composed_g1_contents(
                stable_g1, private / donor.name, unlock_waist_roll=False
            )
        )
        after = Usd.Stage.Open(str(prepared))
        after_hashes, after_materials = _snapshot(after, composed_paths)
        if before_hashes != after_hashes or before_materials != after_materials:
            raise ValueError(
                "Prepared robot differs in composed properties or effective materials."
            )
        receipt["removed_relationships"] = [
            {
                "source_layer": str(source_layers[name]),
                "source_path": source_paths[name],
                "old_target": str(expected_target),
                "new_targets": [],
                "new_relationship_present": False,
                "effective_before": {
                    p: v
                    for p, v in before_materials.items()
                    if p.startswith(f"{G1_STABLE_ROOT_PRIM}/{name}/")
                },
                "effective_after": {
                    p: v
                    for p, v in after_materials.items()
                    if p.startswith(f"{G1_STABLE_ROOT_PRIM}/{name}/")
                },
            }
            for name in links
        ]
        receipt.update(
            composed_before=before_hashes,
            composed_after=after_hashes,
            all_mesh_effective_materials_before=before_materials,
            all_mesh_effective_materials_after=after_materials,
            prepared_robot=str(prepared),
            prepared_robot_sha256=_sha(prepared),
            private_layer_sha256={
                str(path.relative_to(output_dir)): _sha(path)
                for path in sorted(private.rglob("*.usd"))
            },
        )
        receipt["source_sha256_after"] = {path: _sha(Path(path)) for path in inputs}
        if receipt["source_sha256_after"] != inputs:
            raise ValueError("Source assets changed during preparation.")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (output_dir / "preparation.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    return receipt
