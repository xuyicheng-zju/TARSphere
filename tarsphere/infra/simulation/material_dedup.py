"""Build verified material-equivalent USD overlays in a separate USD process.

The source layers remain untouched. Only equal static material networks are
merged; unresolved assets and instance-proxy materials remain independent.
USD imports are local so importing this module never initializes pxr in Kit's
parent process.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

_CACHE_VERSION = "tarsphere.material_dedup.v1"


def _file_stamp(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def overlay_is_current(source: Path, output: Path) -> bool:
    """Validate the generator and dependency cache without importing USD."""
    try:
        marker = json.loads(output.with_name(output.name + ".json").read_text())
        return (
            marker["generator"] == _CACHE_VERSION
            and marker["source"] == str(source.resolve(strict=True))
            and marker["overlay"] == _file_stamp(output)
            and bool(marker["dependencies"])
            and all(
                _file_stamp(Path(path)) == stamp
                for path, stamp in marker["dependencies"].items()
            )
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _file_digest(path: Path, cache: dict[str, str]) -> str:
    key = str(path.resolve(strict=True))
    if key not in cache:
        digest = hashlib.sha256()
        with open(key, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        cache[key] = digest.hexdigest()
    return cache[key]


def _asset_signature(value: Any, attribute: Any, cache: dict[str, str]) -> Any:
    resolved = value.resolvedPath
    if not resolved:
        # USD resolves authored values against the layer that supplies them.
        # The strongest property spec can instead supply only metadata; using
        # its directory to repair an unresolved value would guess the origin.
        raise ValueError(f"USD did not resolve material asset: {value.path}")
    path = Path(resolved)
    if not resolved or not path.is_file():
        raise ValueError(f"Unresolved material asset: {value.path}")
    signature = [path.suffix.lower(), _file_digest(path, cache)]
    if path.suffix.lower() == ".mdl":
        text = path.read_text(encoding="utf-8")
        # Absolute MDL imports share the process search path. Relative module
        # imports can depend on an unenumerated package and stay independent.
        if re.search(r"\b(?:import|using)\s+(?!::)[.\w]", text):
            signature.append(str(path.resolve()))
        resources = []
        for arguments in re.findall(
            r"\b(?:texture_\w+|light_profile|bsdf_measurement)\s*\(([^)]*)\)", text
        ):
            if not arguments.strip():
                continue  # An empty resource constructor has no dependency.
            literal = re.match(r'^\s*"([^"\n]+)"\s*(?:,|$)', arguments)
            if literal is None:
                # Dynamic resource names cannot establish cross-directory
                # equivalence. Retain the module's resolution context.
                signature.append(str(path.resolve()))
            else:
                resources.append(literal.group(1))
        for resource in resources:
            dependency = Path(resource)
            if not dependency.is_absolute():
                dependency = path.parent / dependency
            signature.append([resource, _file_digest(dependency, cache)])
    return signature


def _canonical(value: Any, attribute: Any, cache: dict[str, str]) -> Any:
    from pxr import Sdf

    if isinstance(value, Sdf.AssetPath):
        return {"asset": _asset_signature(value, attribute, cache)}
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonical(item, attribute, cache) for key, item in value.items()
        }
    try:
        return [_canonical(item, attribute, cache) for item in value]
    except TypeError:
        return repr(value)


def _network_signature(material: Any, cache: dict[str, str]) -> str:
    from pxr import Usd

    root = material.GetPath()

    def target(path: Any) -> str:
        return str(path.MakeRelativePath(root)) if path.HasPrefix(root) else str(path)

    network = []
    for prim in Usd.PrimRange(material):
        attributes = []
        for attribute in sorted(prim.GetAttributes(), key=lambda item: item.GetName()):
            if attribute.GetNumTimeSamples() or attribute.ValueMightBeTimeVarying():
                raise ValueError("Animated material remains independent")
            attributes.append(
                [
                    attribute.GetName(),
                    str(attribute.GetTypeName()),
                    _canonical(attribute.Get(), attribute, cache),
                    [target(path) for path in attribute.GetConnections()],
                    _canonical(
                        {
                            key: value
                            for key, value in attribute.GetAllMetadata().items()
                            if key not in {"default", "connectionPaths", "timeSamples"}
                        },
                        attribute,
                        cache,
                    ),
                ]
            )
        relationships = [
            [
                relationship.GetName(),
                [target(path) for path in relationship.GetTargets()],
            ]
            for relationship in sorted(
                prim.GetRelationships(), key=lambda item: item.GetName()
            )
        ]
        network.append(
            [
                str(prim.GetPath().MakeRelativePath(root)),
                prim.GetTypeName(),
                _canonical(prim.GetAllMetadata(), prim, cache),
                attributes,
                relationships,
            ]
        )
    return json.dumps(network, sort_keys=True, allow_nan=False)


def _materials(stage: Any) -> list[Any]:
    from pxr import Usd

    return [
        prim
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if prim.GetTypeName() == "Material"
    ]


def _binding_relationships(prim: Any):
    # Include physics and custom purposes as well as preview/full/default and
    # collection bindings. A material may carry both shading and physics APIs.
    for relationship in prim.GetRelationships():
        name = relationship.GetName()
        if name == "material:binding" or name.startswith("material:binding:"):
            yield relationship


def _externally_referenced(stage: Any, candidates: set[str]) -> set[str]:
    from pxr import Usd

    held = set()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        targets = [path for rel in prim.GetRelationships() for path in rel.GetTargets()]
        targets.extend(
            path for attr in prim.GetAttributes() for path in attr.GetConnections()
        )
        for path in targets:
            owner = path.GetPrimPath()
            while not owner.isEmpty and not owner.IsAbsoluteRootPath():
                if str(owner) in candidates:
                    if not prim.GetPath().HasPrefix(owner):
                        held.add(str(owner))
                    break
                owner = owner.GetParentPath()
    return held


def _verify(stage: Any, expected_materials: int) -> None:
    from pxr import Usd

    if len(_materials(stage)) != expected_materials:
        raise RuntimeError("Material overlay has an unexpected active material count")
    for material in _materials(stage):
        for node in Usd.PrimRange(material, Usd.TraverseInstanceProxies()):
            for attribute in node.GetAttributes():
                for path in attribute.GetConnections():
                    connected = stage.GetPropertyAtPath(path)
                    if not connected or not connected.GetPrim().IsActive():
                        raise RuntimeError(
                            f"Dangling material connection: {attribute.GetPath()} -> {path}"
                        )
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        for relationship in _binding_relationships(prim):
            for path in relationship.GetTargets():
                bound = stage.GetPrimAtPath(path.GetPrimPath())
                if not bound or not bound.IsActive():
                    raise RuntimeError(
                        f"Dangling material binding: {relationship.GetPath()} -> {path}"
                    )


def bake_material_overlay(source: Path, output: Path) -> dict[str, int]:
    """Publish one verified overlay atomically, retaining the original scene."""
    from pxr import Ar, Sdf, Usd

    source, output = source.resolve(strict=True), output.resolve()
    if source == output:
        raise ValueError("Material deduplication cannot overwrite the source scene")
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Cannot open source scene: {source}")
    groups = defaultdict(list)
    cache = {}
    materials = _materials(stage)
    with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
        for material in materials:
            if material.IsInstanceProxy() or material.IsInPrototype():
                continue
            try:
                signature = _network_signature(material, cache)
            except (ValueError, OSError, UnicodeError):
                continue  # Unproven equivalence never changes a material.
            groups[signature].append(str(material.GetPath()))
    remap = {}
    for paths in groups.values():
        canonical, *duplicates = sorted(paths)
        remap.update((duplicate, canonical) for duplicate in duplicates)
    temporary = output.with_name(f".{output.stem}-{uuid4().hex}{output.suffix}")
    try:
        overlay = Usd.Stage.CreateNew(str(temporary))
        overlay.GetRootLayer().subLayerPaths = [os.path.relpath(source, output.parent)]
        for name in (
            "upAxis",
            "metersPerUnit",
            "timeCodesPerSecond",
            "framesPerSecond",
            "startTimeCode",
            "endTimeCode",
            "defaultPrim",
        ):
            if stage.HasAuthoredMetadata(name):
                overlay.SetMetadata(name, stage.GetMetadata(name))
        rewritten = 0
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if prim.IsInstanceProxy():
                continue
            for relationship in _binding_relationships(prim):
                targets = relationship.GetTargets()
                updated = [
                    Sdf.Path(remap.get(str(path), str(path))) for path in targets
                ]
                if updated != targets:
                    over = overlay.OverridePrim(prim.GetPath())
                    over.CreateRelationship(
                        relationship.GetName(), custom=False
                    ).SetTargets(updated)
                    rewritten += 1
        overlay.GetRootLayer().Save()
        composed = Usd.Stage.Open(str(temporary))
        held = _externally_referenced(composed, set(remap))
        for duplicate in sorted(set(remap) - held):
            overlay.OverridePrim(duplicate).SetActive(False)
        overlay.GetRootLayer().Save()
        composed.Reload()
        deactivated = len(set(remap) - held)
        _verify(composed, len(materials) - deactivated)
        temporary.replace(output)
        dependencies = {str(source), *cache}
        dependencies.update(
            layer.realPath for layer in stage.GetUsedLayers() if layer.realPath
        )
        marker = {
            "generator": _CACHE_VERSION,
            "source": str(source),
            "overlay": _file_stamp(output),
            "dependencies": {
                str(Path(path).resolve()): _file_stamp(Path(path))
                for path in sorted(dependencies)
            },
        }
        marker_path = output.with_name(output.name + ".json")
        marker_temporary = marker_path.with_name(
            marker_path.name + ".tmp-" + uuid4().hex
        )
        try:
            marker_temporary.write_text(json.dumps(marker, sort_keys=True) + "\n")
            marker_temporary.replace(marker_path)
        finally:
            marker_temporary.unlink(missing_ok=True)
        return {
            "materials": len(materials),
            "deactivated": deactivated,
            "rewritten_bindings": rewritten,
        }
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(bake_material_overlay(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
