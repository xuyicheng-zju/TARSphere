"""GRScenes MDL module roots for Isaac / Kit neuray.

USD shaders in both home and commercial collections author

    info:mdl:sourceAsset = @./Materials/Num….mdl@

Kit's MDLC turns that into module ``::.::Materials::Num…`` and searches
``MDL_SYSTEM_PATH`` (not the USD resolved file path).  Putting only the
``Materials`` directory on that path makes neuray look for
``Materials/Materials/Num….mdl``, which does not exist.  Commercial RGB BEVs
then log ``C120 could not find module`` and RTX paints the missing-shader red
fallback (~RGB 220, 58, 58).

The collection (or scene) directory that *contains* ``Materials/`` must be on
the system path so ``Materials::Num…`` resolves.  ``Materials`` itself stays
on the path so top-level ``::OmniUe4Function`` / ``::Num…`` still work.

``MDL_USER_PATH`` is intentionally not used: Kit's material-graph registry
blocks user-path modules by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

# NVIDIA vMaterials / Twinbru drapery are authored as http(s) sourceAssets.
# ``https:/host`` (one slash) is the Twinbru form in GRScenes metadata.
_REMOTE_ASSET_PREFIXES = ("http:/", "https:/", "omniverse://")

# Carb list that omni.mdl.usd_converter and neuray also honor, in addition to
# the MDL_SYSTEM_PATH environment variable snapped at SimulationApp start.
_ADDITIONAL_SYSTEM_PATHS_SETTING = "/app/mdl/additionalSystemPaths"

_ACTIVE_PATHS: list[str] = []


def _dedupe(paths: Iterable[Path | str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in paths:
        key = str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def resolve_kit_stock_mdl_search_paths() -> list[Path]:
    """NVIDIA / Omni built-in MDL roots (OmniPBR, OmniGlass, ``nvidia::``).

    Named GRScenes modules such as ``Ground_Leaves`` / ``Tinted_Glass_R50``
    are vMaterials identifiers.  The files are not in Kit, but stand-ins
    ``import ::OmniPBR`` / ``::OmniGlass`` / ``::nvidia::core_definitions``
    and those *do* live here.  Home Materials is never added (no mix-mount).
    """
    roots: list[Path] = []
    isaac = os.environ.get("ISAAC_SIM_PATH", "").strip()
    if isaac:
        base = Path(isaac)
        for rel in (
            "kit/mdl/core/Base",
            "kit/mdl/core/mdl",
            "kit/mdl/core/Ue4",
            "kit/resources/materials",
        ):
            candidate = (base / rel).resolve()
            if candidate.is_dir():
                roots.append(candidate)
    extra = os.environ.get("TARSPHERE_MDL_STOCK_PATHS", "")
    for item in extra.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            roots.append(path.resolve())
    for env_name in ("TARSPHERE_ASSETS_PATH",):
        assets = os.environ.get(env_name, "").strip()
        if not assets:
            continue
        for rel in ("materials", "materials/vMaterials_2", "vMaterials_2"):
            candidate = (Path(assets) / rel).resolve()
            if candidate.is_dir():
                roots.append(candidate)
                if candidate.name == "vMaterials_2" and candidate.parent.is_dir():
                    roots.append(candidate.parent)
    return [Path(item) for item in _dedupe(roots)]


def resolve_grscenes_mdl_search_paths(scene_path: Path) -> list[Path]:
    """Return module roots so both ``Materials::X`` and ``::X`` resolve.

    ``scene_path`` may be a USD file or a GRScene directory.  The sibling
    ``Materials`` link is resolved; the scene directory, the resolved
    collection root, and the Materials directory itself are all included.
    Kit stock roots are added by ``configure_mdl_system_path``, not here,
    so collection trees stay separate from the home twin.
    """
    scene_dir = scene_path if scene_path.is_dir() else scene_path.parent
    materials = (scene_dir / "Materials").resolve()
    if not materials.is_dir():
        return []

    roots: list[Path] = []
    if scene_dir.exists():
        roots.append(scene_dir.resolve())
    parent = materials.parent
    if parent.is_dir():
        roots.append(parent)
    roots.append(materials)
    return [Path(item) for item in _dedupe(roots)]


def configure_mdl_system_path(
    scene_path: Path,
    *,
    warn: bool = True,
) -> list[str]:
    """Prepend GRScenes + Kit stock MDL roots to ``MDL_SYSTEM_PATH``.

    Used by every Isaac entrypoint (BEV, ``IsaacAutonomousEnv``, smokes)
    before SimulationApp so ``::OmniPBR`` / ``::OmniGlass`` stand-ins and
    ``Materials::Num…`` both compile.
    """
    global _ACTIVE_PATHS

    resolved = resolve_grscenes_mdl_search_paths(scene_path)
    stock = resolve_kit_stock_mdl_search_paths()
    if not resolved:
        if warn and "GRScenes" in str(scene_path):
            materials = (
                (scene_path / "Materials")
                if scene_path.is_dir()
                else (scene_path.parent / "Materials")
            )
            print(
                f"[WARN] GRScenes Materials directory not found: {materials}. "
                "Camera RGB may render with incomplete materials.",
                flush=True,
            )
        if not stock:
            return list(_ACTIVE_PATHS)

    added = [str(path) for path in [*resolved, *stock]]
    existing = [
        item for item in os.environ.get("MDL_SYSTEM_PATH", "").split(os.pathsep) if item
    ]
    os.environ["MDL_SYSTEM_PATH"] = os.pathsep.join(_dedupe([*added, *existing]))
    _ACTIVE_PATHS = _dedupe([*added, *_ACTIVE_PATHS])
    if resolved:
        print(
            "[INFO] MDL_SYSTEM_PATH includes GRScenes roots: "
            + ", ".join(str(path) for path in resolved),
            flush=True,
        )
    if stock:
        print(
            "[INFO] MDL_SYSTEM_PATH includes Kit stock MDL roots: "
            + ", ".join(str(path) for path in stock),
            flush=True,
        )
    return list(_ACTIVE_PATHS)


def additional_system_path_extra_args(
    paths: Sequence[str] | None = None,
) -> list[str]:
    """Kit CLI appends for ``/app/mdl/additionalSystemPaths``."""
    items = list(paths) if paths is not None else list(_ACTIVE_PATHS)
    return [f"--{_ADDITIONAL_SYSTEM_PATHS_SETTING}+={path}" for path in items]


def _asset_path_text(value: object) -> str:
    path = getattr(value, "path", None)
    return str(path if path is not None else value)


def _layer_filesystem_path(layer: object) -> Path | None:
    for attr in ("realPath", "identifier"):
        value = getattr(layer, attr, None)
        if not value:
            continue
        text = str(value)
        if text.startswith("anon:") or "://" in text:
            continue
        path = Path(text)
        if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
            return path
    return None


def _is_asset_attribute(attr: object) -> bool:
    type_name = str(getattr(attr, "typeName", "") or "")
    if type_name == "asset":
        return True
    return hasattr(getattr(attr, "default", None), "path")


def is_remote_material_asset(path: str) -> bool:
    """True for http(s) / omniverse material assets, including ``https:/host``."""
    text = str(path).strip().replace("\\", "/").lower()
    return text.startswith(_REMOTE_ASSET_PREFIXES)


def remote_mdl_basename(path: str) -> str | None:
    """Basename of a remote ``.mdl`` asset, or None if the path is not one."""
    text = str(path).strip().replace("\\", "/").split("?", 1)[0]
    if not is_remote_material_asset(text):
        return None
    name = Path(text).name
    if not name.endswith(".mdl"):
        return None
    return name


def normalize_relative_mdl_source_asset(path: str) -> str | None:
    """Drop the ``./`` prefix that makes MDLC search ``::.::Materials::…``.

    ``@./Materials/Num….mdl@`` becomes module ``::.::Materials::Num…``, which is
    resolved only against the USD layer directory and does not consult
    ``MDL_SYSTEM_PATH``.  The same file authored as ``@Materials/Num….mdl@``
    (the WorldGrid form that already compiles) becomes ``::Materials::Num…``
    and is found in the collection root we put on the system path.
    """
    text = str(path).strip()
    if not text.startswith("./"):
        return None
    return text[2:]


def normalize_remote_mdl_source_asset(
    path: str,
    *,
    materials_dir: Path | None,
) -> str | None:
    """Map HTTP / omniverse MDL sourceAssets onto a local collection module.

    GRScenes authors NVIDIA vMaterials and Twinbru drapery as
    ``@http://omniverse-content-production.s3…/Ash_Planks.mdl@`` and a broken
    ``@https:/twinbru…/Graphical_White_Jacquard_FL_3075963.mdl@``.  Kit fetches
    the remote module, then queues its S3 textures / 404 thumbs; those nine
    files never idle.  Local Kit-backed stand-ins already live at
    ``Materials/<basename>.mdl``.  Only rewrite when that file exists.
    """
    basename = remote_mdl_basename(path)
    if basename is None:
        return None
    rewritten = f"Materials/{basename}"
    if materials_dir is None:
        return rewritten
    dest = materials_dir / basename
    if dest.is_file():
        return rewritten
    return None


def normalize_relative_material_texture_asset(
    path: str,
    *,
    layer_path: Path | None,
    materials_dir: Path | None,
) -> str | None:
    """Rewrite collection-relative texture assets so Hydra can find them.

    GRScenes shaders author ``inputs:*_Tex = @./Materials/Textures/foo.png@``.
    USD resolves ``./`` against the *layer* directory (``models/.../instance.usd``),
    not the collection root, so Hydra looks for
    ``models/.../Materials/Textures/foo.png`` forever.  The files live at
    ``<collection>/Materials/Textures/foo.png``.

    Stripping ``./`` from ``info:mdl:sourceAsset`` is not enough: UsdToMdl then
    still receives the leftover ``./Materials/Textures`` inputs and the working
    set never goes idle.  This rewrites those texture assets to a path
    relative to the layer that lands on the collection Materials tree.
    """
    text = str(path).strip().replace("\\", "/")
    if text.startswith("./"):
        stripped = text[2:]
    else:
        stripped = text
    if not stripped.startswith("Materials/"):
        return None
    if stripped.endswith(".mdl"):
        return None
    if materials_dir is None:
        return stripped if text.startswith("./") else None
    collection_root = materials_dir.resolve().parent
    target = (collection_root / stripped).resolve()
    if layer_path is None:
        return str(target)
    try:
        rewritten = Path(
            os.path.relpath(target, layer_path.parent.resolve())
        ).as_posix()
    except ValueError:
        rewritten = str(target)
    if rewritten == text:
        return None
    return rewritten


def rewrite_relative_mdl_source_assets_in_layer(
    layer: object,
    *,
    materials_dir: Path | None = None,
) -> int:
    """Rewrite leftover ``./Materials/…`` MDL and texture assets on one layer."""
    from pxr import Sdf

    changed = 0
    layer_path = _layer_filesystem_path(layer)

    def visit(prim_spec: object) -> None:
        nonlocal changed
        attributes = getattr(prim_spec, "attributes", None)
        if attributes is not None:
            for name, attr in attributes.items():
                if (
                    attr is None
                    or attr.default is None
                    or not _is_asset_attribute(attr)
                ):
                    continue
                current = _asset_path_text(attr.default)
                if name == "info:mdl:sourceAsset":
                    rewritten = normalize_relative_mdl_source_asset(current)
                    if rewritten is None:
                        rewritten = normalize_remote_mdl_source_asset(
                            current,
                            materials_dir=materials_dir,
                        )
                else:
                    rewritten = normalize_relative_material_texture_asset(
                        current,
                        layer_path=layer_path,
                        materials_dir=materials_dir,
                    )
                if rewritten is not None:
                    attr.default = Sdf.AssetPath(rewritten)
                    changed += 1
        for child in getattr(prim_spec, "nameChildren", ()):
            visit(child)

    visit(layer.pseudoRoot)
    return changed


def rewrite_relative_mdl_source_assets_on_disk(
    usd_path: Path,
    *,
    save: bool = True,
    materials_dir: Path | None = None,
) -> int:
    """Open one USD layer, rewrite leftover ``./Materials`` assets, optionally save."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        raise FileNotFoundError(f"Cannot open USD layer: {usd_path}")
    changed = rewrite_relative_mdl_source_assets_in_layer(
        layer,
        materials_dir=materials_dir,
    )
    if changed and save:
        if not layer.Save():
            raise RuntimeError(f"Failed to save rewritten material assets: {usd_path}")
    return changed
