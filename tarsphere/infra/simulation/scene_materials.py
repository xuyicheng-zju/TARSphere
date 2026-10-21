"""Single GRScenes material contract for every Isaac entrypoint.

BEV generation and ``IsaacAutonomousEnv`` must both:

1. recover any ``.mdl`` / texture the opened collection is missing from a
   sibling GRScenes Materials tree (home twin, or an explicit donor);
2. put the collection root on ``MDL_SYSTEM_PATH`` so ``Materials::Num…``
   compiles;
3. load the same verified material-dedup overlay the owned baker writes;
4. rewrite leftover ``@./Materials/…@`` sourceAssets *and* ``inputs:*_Tex``
   texture assets so Hydra does not look for ``models/…/Materials/Textures``;
   also rewrite HTTP / ``https:/`` vMaterials sourceAssets onto local
   ``Materials/<basename>.mdl`` stand-ins *before* Kit opens, or the working
   set parks on remote S3 textures and 404 thumbs;
5. wait until Kit's stage-loading working set is *idle* before the first
   RGB capture — not 50% of files, which leaves furniture black / plastic
   white / error-red.

Home and commercial stay separate trees.  Missing commercial modules are
*copied into* ``commercial_scenes/Materials`` (CONSTRAINT: do not mix-mount
the two collections on the search path).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

from tarsphere.infra.simulation.mdl_search_paths import (
    _asset_path_text,
    configure_mdl_system_path,
    is_remote_material_asset,
    remote_mdl_basename,
    resolve_kit_stock_mdl_search_paths,
    rewrite_relative_mdl_source_assets_in_layer,
    rewrite_relative_mdl_source_assets_on_disk,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MDL_REF_RE = re.compile(rb"(?:\./)?Materials[/\\]([A-Za-z0-9_.-]+\.mdl)")
# Crate (binary .usd) often intern-splits ``Materials/Num….mdl`` so the
# contiguous path is gone.  The filename token and the Num hash still appear.
# Named vMaterials (Ground_Leaves, Tinted_Glass_R50, …) are a second class:
# they are not ``Num<24hex>`` and commercial Materials does not ship them.
_MDL_NAME_RE = re.compile(
    rb"((?:Num[0-9a-fA-F]{24}|WorldGridMaterial|OmniUe4[A-Za-z]+|"
    rb"DayMaterial|KooPbr(?:_maps)?|"
    rb"[A-Z][A-Za-z0-9]{2,}(?:_[A-Za-z0-9]+)+)\.mdl)"
)
# GRScenes material ids are ``Num`` + 24 hex (Mongo ObjectId).  Shorter
# matches are crate string-table fragments, not missing modules.
_NUM_HASH_RE = re.compile(rb"Num[0-9a-fA-F]{24}")
_HEX_ID_MDL_RE = re.compile(r"^(?:Num|MI_)?[0-9a-fA-F]{20,}\.mdl$")
_NAMED_MDL_STEM_RE = re.compile(r"^(?:[A-Z][A-Za-z0-9]{2,}(?:_[A-Za-z0-9]+)+|Mortar)$")
_TEXTURE_REF_RE = re.compile(r"""["']((?:\./)?Textures/[^"']+)["']""")
# Stage-proven named modules from the parked commercial batch.  Crate layers
# intern-split ``Materials/Ground_Leaves.mdl``, so a raw-bytes path scan
# misses them; the stem still often survives as a string-table token.
_KNOWN_NAMED_MDL = (
    "Ash_Planks.mdl",
    "Glass_Glazing_Spandrel.mdl",
    "Graphical_White_Jacquard_FL_3075963.mdl",
    "Ground_Leaves.mdl",
    "Mortar.mdl",
    "Retroreflective_Material.mdl",
    "Tinted_Glass_R50.mdl",
    "Tinted_Glass_R75.mdl",
)
# Kit-backed stand-ins used only when zip / home / Kit / vMaterials lack the
# file.  ``kind`` is pbr / glass / retro.  Colors are plausible AEC defaults.
_STOCK_NAMED_MODULES: dict[str, tuple[str, tuple[float, float, float], float]] = {
    "Ash_Planks.mdl": ("pbr", (0.62, 0.52, 0.38), 0.55),
    "Glass_Glazing_Spandrel.mdl": ("glass", (0.38, 0.42, 0.44), 1.50),
    "Graphical_White_Jacquard_FL_3075963.mdl": ("pbr", (0.92, 0.91, 0.88), 0.65),
    "Ground_Leaves.mdl": ("pbr", (0.27, 0.33, 0.14), 0.82),
    "Mortar.mdl": ("pbr", (0.72, 0.70, 0.66), 0.88),
    "Retroreflective_Material.mdl": ("retro", (0.85, 0.85, 0.80), 0.20),
    "Tinted_Glass_R50.mdl": ("glass", (0.45, 0.62, 0.72), 1.52),
    "Tinted_Glass_R75.mdl": ("glass", (0.25, 0.40, 0.55), 1.52),
}
# A vMaterials module filename and the subIdentifier authored by USD are not
# always the same string.  Export every stage-observed name from synthesized
# stand-ins so the local module satisfies the original shader contract.
_STOCK_NAMED_EXPORTS: dict[str, tuple[str, ...]] = {
    "Ground_Leaves.mdl": ("Ground_Leaves", "ground_leaves_green"),
}
_REWRITE_CODE = r"""
import sys
from pathlib import Path
from tarsphere.infra.simulation.scene_materials import rewrite_scene_material_asset_paths

count = rewrite_scene_material_asset_paths(Path(sys.argv[1]))
print(f'REWRITE_COUNT {count}')
"""
_SDF_LIST_CODE = r"""
import sys
from pathlib import Path
from pxr import Sdf

names = set()

def visit(spec):
    attrs = getattr(spec, "attributes", None)
    if attrs is not None:
        attr = attrs.get("info:mdl:sourceAsset")
        if attr is not None and attr.default is not None:
            value = attr.default
            text = str(getattr(value, "path", value) or "")
            base = Path(text).name
            if base.endswith(".mdl"):
                names.add(base)
    for child in getattr(spec, "nameChildren", ()):
        visit(child)

for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    layer = Sdf.Layer.FindOrOpen(path)
    if layer is None:
        continue
    root = getattr(layer, "pseudoRoot", None)
    if root is not None:
        visit(root)
sys.stdout.write("\n".join(sorted(names)))
"""
# Kit reports (is_loading, files_loaded, total_files).  Idle is 0 / 0 after
# the working set has been observed.  Capturing at 50% is how AEI8 kept
# black shelf clutter and plastic-white sofas after the ./Materials rewrite.
_DEFAULT_WAIT_MAX_UPDATES = 4800
_DEFAULT_WAIT_MAX_SEC = 600.0
_DEFAULT_CONFIRMATION_FRAMES = 5
_DEFAULT_MIN_AVAILABLE_GIB = 3.0


def _env_flag(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def collection_root_for(scene_path: Path) -> Path | None:
    """Return the GRScenes collection directory that owns ``scene_path``."""
    current = scene_path if scene_path.is_dir() else scene_path.parent
    current = current.resolve()
    for parent in (current, *current.parents):
        if (
            parent.name in {"home_scenes", "commercial_scenes"}
            and (parent / "Materials").is_dir()
        ):
            return parent
    materials = (
        (scene_path / "Materials")
        if scene_path.is_dir()
        else (scene_path.parent / "Materials")
    ).resolve()
    if materials.is_dir():
        return materials.parent
    return None


def materials_dir_for(scene_path: Path) -> Path | None:
    root = collection_root_for(scene_path)
    if root is None:
        return None
    materials = (root / "Materials").resolve()
    return materials if materials.is_dir() else None


def localize_referenced_asset_materials(prim: object, asset_path: Path) -> int:
    """Resolve a late-injected asset's shaders in its own source collection.

    Cold scene preparation cannot see future episode references. Author only
    path overrides under the injected prim, leaving source USD, colours and
    surrounding scene materials unchanged. Absolute assets also avoid resolving
    another collection's identically named module through the active scene.
    """
    from pxr import Sdf, Usd, UsdShade

    materials = materials_dir_for(asset_path)
    changed = 0
    for shader_prim in Usd.PrimRange(prim):
        if not shader_prim.IsA(UsdShade.Shader):
            continue
        for attr in shader_prim.GetAttributes():
            value = attr.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.path:
                continue
            path = value.path.replace("\\", "/")
            marker = "Materials/"
            if marker not in path:
                continue
            resolved = Path(value.resolvedPath) if value.resolvedPath else None
            if resolved is None or not resolved.is_file():
                if materials is None:
                    raise FileNotFoundError(
                        f"No material collection for injected asset {asset_path}"
                    )
                resolved = materials / path.split(marker, 1)[1]
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Injected material asset is missing: {asset_path}: {path}"
                )
            absolute = str(resolved.resolve())
            if path != absolute:
                if not attr.Set(Sdf.AssetPath(absolute)):
                    raise RuntimeError(
                        f"Cannot localize injected material asset: {attr.GetPath()}"
                    )
                changed += 1
    return changed


def sibling_material_donors(scene_path: Path) -> list[Path]:
    """Other GRScenes Materials trees that may hold missing modules.

    Never includes ``/mnt`` or ``/tmp`` unless the caller listed them in
    ``TARSPHERE_GRSCENES_MATERIAL_DONORS``.  The home twin is the default donor
    for commercial, and the reverse, because both collections share the
    same ``Num<hash>.mdl`` identifiers.
    """
    donors: list[Path] = []
    root = collection_root_for(scene_path)
    if root is not None and root.parent.name == "GRScenes-100":
        sibling_name = (
            "home_scenes" if root.name == "commercial_scenes" else "commercial_scenes"
        )
        sibling = (root.parent / sibling_name / "Materials").resolve()
        if sibling.is_dir():
            donors.append(sibling)
    extra = _env_flag(
        "TARSPHERE_GRSCENES_MATERIAL_DONORS", "TARSPHERE_GRSCENES_MATERIAL_DONORS"
    )
    for item in extra.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            donors.append(path.resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for donor in donors:
        key = str(donor)
        if key in seen:
            continue
        seen.add(key)
        unique.append(donor)
    return unique


def stock_material_donors() -> list[Path]:
    """Kit / vMaterials directories that may hold named Omni modules.

    Copy sources only.  These roots are *not* a substitute for putting the
    collection Materials tree on ``MDL_SYSTEM_PATH``.
    """
    donors: list[Path] = []
    for path in resolve_kit_stock_mdl_search_paths():
        if path.is_dir():
            donors.append(path)
    extra = _env_flag("TARSPHERE_MDL_STOCK_PATHS", "TARSPHERE_MDL_STOCK_PATHS")
    for item in extra.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            donors.append(path.resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for donor in donors:
        key = str(donor)
        if key in seen:
            continue
        seen.add(key)
        unique.append(donor)
    return unique


def _iter_metadata_instance_usd(scene_path: Path) -> list[Path]:
    scene_dir = scene_path if scene_path.is_dir() else scene_path.parent
    meta = scene_dir / "metadata.json"
    root = collection_root_for(scene_path)
    if not meta.is_file() or root is None:
        return []
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    models = payload.get("models") or []
    found: list[Path] = []
    for rel in models:
        path = root / str(rel)
        if path.is_file():
            found.append(path)
    return found


def _known_mdl_basenames(scene_path: Path) -> set[str]:
    known: set[str] = set(_KNOWN_NAMED_MDL)
    materials = materials_dir_for(scene_path)
    if materials is not None:
        known.update(path.name for path in materials.glob("*.mdl"))
    for donor in [*sibling_material_donors(scene_path), *stock_material_donors()]:
        known.update(path.name for path in donor.glob("*.mdl"))
    return known


def _normalize_mdl_name(name: str) -> str:
    base = Path(str(name).strip()).name
    if base and not base.endswith(".mdl"):
        base = f"{base}.mdl"
    return base


def _is_hex_module(name: str) -> bool:
    return bool(_HEX_ID_MDL_RE.match(name))


def _is_named_module(name: str) -> bool:
    if name in _STOCK_NAMED_MODULES or name in _KNOWN_NAMED_MDL:
        return True
    if _is_hex_module(name):
        return False
    stem = name[:-4] if name.endswith(".mdl") else name
    return bool(_NAMED_MDL_STEM_RE.match(stem))


def _mdl_basenames_from_usd_bytes(data: bytes, *, known: set[str]) -> set[str]:
    names = {match.decode("ascii", "ignore") for match in _MDL_REF_RE.findall(data)}
    names.update(
        match.decode("ascii", "ignore") for match in _MDL_NAME_RE.findall(data)
    )
    for match in _NUM_HASH_RE.findall(data):
        candidate = match.decode("ascii", "ignore") + ".mdl"
        if candidate in known:
            names.add(candidate)
    for name in known:
        if not _is_named_module(name):
            continue
        stem = name[:-4] if name.endswith(".mdl") else name
        token = stem.encode("ascii", "ignore")
        if len(token) >= 6 and token in data:
            names.add(name if name.endswith(".mdl") else f"{name}.mdl")
    return names


def _mdl_basenames_via_sdf_subprocess(scene_path: Path) -> set[str]:
    """Read ``info:mdl:sourceAsset`` via a child process so prepare() never imports pxr."""
    layers = _scene_layer_candidates(scene_path)
    if not layers:
        return set()
    try:
        result = subprocess.run(
            [sys.executable, "-c", _SDF_LIST_CODE],
            input="\n".join(str(path) for path in layers),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(
                _env_flag("TARSPHERE_SDF_MDL_INVENTORY_TIMEOUT", default="45")
            ),
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    names: set[str] = set()
    for line in (result.stdout or "").splitlines():
        name = _normalize_mdl_name(line)
        if name.endswith(".mdl"):
            names.add(name)
    return names


def _mdl_basenames_from_metadata(scene_path: Path) -> set[str]:
    """MDL basenames from ``metadata.json`` materials, including HTTP URLs.

    Crate layers intern-split ``http://…/Ash_Planks.mdl`` so a raw-bytes scan
    misses the named module.  The scene metadata keeps the full URL.
    """
    scene_dir = scene_path if scene_path.is_dir() else scene_path.parent
    meta = scene_dir / "metadata.json"
    if not meta.is_file():
        return set()
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    names: set[str] = set()
    materials = payload.get("materials") or []
    if isinstance(materials, dict):
        materials = list(materials.values()) + list(materials.keys())
    for item in materials:
        text = str(item).strip().replace("\\", "/")
        remote = remote_mdl_basename(text)
        if remote is not None:
            names.add(remote)
            continue
        name = Path(text.split("?", 1)[0]).name
        if name.endswith(".mdl"):
            names.add(name)
    return names


def referenced_mdl_basenames(
    scene_path: Path,
    *,
    extra_referenced: Iterable[str] | None = None,
) -> set[str]:
    """MDL basenames authored on the scene's model ``instance.usd`` layers."""
    names: set[str] = set()
    known = _known_mdl_basenames(scene_path)
    for path in _iter_metadata_instance_usd(scene_path):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        names.update(_mdl_basenames_from_usd_bytes(data, known=known))
        # Never import pxr here. prepare() runs before SimulationApp; loading
        # conda usd-core first makes Kit SIGSEGV (Tf_PyEnumWrapper / rc=139).
    names.update(_mdl_basenames_from_metadata(scene_path))
    if _truthy(_env_flag("TARSPHERE_SDF_MDL_INVENTORY", default="")):
        names.update(_mdl_basenames_via_sdf_subprocess(scene_path))
    if extra_referenced is not None:
        names.update(_normalize_mdl_name(item) for item in extra_referenced if item)
    return {name for name in names if name.endswith(".mdl")}


def _texture_refs_in_mdl(mdl_path: Path) -> set[str]:
    try:
        text = mdl_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return set()
    refs: set[str] = set()
    for match in _TEXTURE_REF_RE.findall(text):
        refs.add(match[2:] if match.startswith("./") else match)
    return refs


def _copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _stock_named_mdl_source(name: str) -> str | None:
    """Kit-backed stand-in for a named vMaterials module that no donor has."""
    spec = _STOCK_NAMED_MODULES.get(name)
    stem = name[:-4] if name.endswith(".mdl") else name
    if spec is None:
        if not _is_named_module(name):
            return None
        spec = ("pbr", (0.55, 0.55, 0.52), 0.6)
    kind, rgb, roughness = spec
    r, g, b = rgb
    export_names = _STOCK_NAMED_EXPORTS.get(name, (stem,))
    if kind == "glass":
        definitions = "\n\n".join(
            f"export material {export_name}()\n"
            "= ::OmniGlass::OmniGlass(\n"
            f"    glass_color: color({r:.3f}, {g:.3f}, {b:.3f})\n"
            ");"
            for export_name in export_names
        )
        return "mdl 1.7;\n\nimport ::OmniGlass::*;\n\n" + definitions + "\n"
    if kind == "retro":
        definitions = "\n\n".join(
            f"export material {export_name}()\n= retroreflective();"
            for export_name in export_names
        )
        return (
            "mdl 1.7;\n\nimport ::nvidia::core_definitions::*;\n\n" + definitions + "\n"
        )
    definitions = "\n\n".join(
        f"export material {export_name}()\n"
        "= ::OmniPBR::OmniPBR(\n"
        f"    diffuse_color_constant: color({r:.3f}, {g:.3f}, {b:.3f}),\n"
        f"    reflection_roughness_constant: {roughness:.3f}\n"
        ");"
        for export_name in export_names
    )
    return "mdl 1.7;\n\nimport ::OmniPBR::*;\n\n" + definitions + "\n"


def _write_stock_named_mdl(dest: Path) -> bool:
    text = _stock_named_mdl_source(dest.name)
    if text is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return True


def ensure_stock_named_modules(
    materials: Path,
    names: Iterable[str] | None = None,
    *,
    extra_donors: Iterable[Path] | None = None,
) -> list[str]:
    """Copy or synthesize named Omni modules into a collection Materials dir."""
    donors = [*stock_material_donors()]
    if extra_donors is not None:
        donors.extend(
            Path(item).resolve() for item in extra_donors if Path(item).is_dir()
        )
    written: list[str] = []
    for raw in names if names is not None else _KNOWN_NAMED_MDL:
        name = _normalize_mdl_name(raw)
        dest = materials / name
        if dest.is_file():
            continue
        source = _find_on_donors(name, donors)
        if source is not None:
            _copy_file(source, dest)
            written.append(name)
            continue
        if _write_stock_named_mdl(dest):
            written.append(name)
    return written


def _find_on_donors(rel: str, donors: Iterable[Path]) -> Path | None:
    rel_path = Path(rel)
    basename = rel_path.name
    for donor in donors:
        candidate = donor / rel
        if candidate.is_file():
            return candidate
        if basename and (donor / "Textures" / basename).is_file():
            return donor / "Textures" / basename
        if (
            "/" not in rel.replace("\\", "/")
            and _is_named_module(basename)
            and any(part.lower().startswith("vmaterial") for part in donor.parts)
        ):
            try:
                matches = [path for path in donor.rglob(basename) if path.is_file()]
            except OSError:
                matches = []
            if matches:
                return matches[0]
    return None


def recover_missing_collection_materials(
    scene_path: Path,
    *,
    extra_donors: Iterable[Path] | None = None,
    extra_referenced: Iterable[str] | None = None,
    raise_if_missing: bool = True,
) -> dict[str, object]:
    """Copy missing ``.mdl`` / textures into this collection from donors.

    Named Omni / vMaterials modules (``Ground_Leaves``, glass, planks, …)
    are recovered from the home twin, an official zip, Kit/vMaterials stock,
    or — if none of those have the file — a Kit-backed stand-in that imports
    ``OmniPBR`` / ``OmniGlass`` / ``nvidia::core_definitions``.  Hex
    ``Num<24>`` modules still require a real donor file.

    Raises if a referenced module still cannot be found after every donor
    (unless ``raise_if_missing`` is false).  That is deliberate: shipping
    RTX error-red / plastic-white is not a successful load.
    """
    materials = materials_dir_for(scene_path)
    if materials is None:
        return {
            "recovered_mdl": [],
            "recovered_textures": [],
            "referenced_mdl": [],
            "still_missing_mdl": [],
            "synthesized_mdl": [],
        }

    present = {path.name for path in materials.glob("*.mdl")}
    referenced = sorted(
        referenced_mdl_basenames(scene_path, extra_referenced=extra_referenced)
    )
    donors = [
        *sibling_material_donors(scene_path),
        *stock_material_donors(),
    ]
    if extra_donors is not None:
        donors.extend(
            Path(item).resolve() for item in extra_donors if Path(item).is_dir()
        )

    recovered_mdl: list[str] = []
    recovered_textures: list[str] = []
    synthesized_mdl: list[str] = []
    still_missing: list[str] = []

    needed = [name for name in referenced if name not in present]
    # Texture completeness is required for modules this scene actually names,
    # including ones we just recovered.  Do not fail the collection because an
    # unused leftover .mdl is missing a map.
    used_mdls = set(referenced)
    for name in needed:
        source = _find_on_donors(name, donors)
        dest = materials / name
        if source is not None:
            _copy_file(source, dest)
            recovered_mdl.append(name)
            present.add(name)
            used_mdls.add(name)
            continue
        if _write_stock_named_mdl(dest):
            synthesized_mdl.append(name)
            recovered_mdl.append(name)
            present.add(name)
            used_mdls.add(name)
            print(
                f"[INFO] Synthesized Kit-backed named MDL {name} into {dest}",
                flush=True,
            )
            continue
        still_missing.append(name)
    mdl_to_scan = [
        materials / name for name in sorted(used_mdls) if (materials / name).is_file()
    ]

    missing_textures: list[str] = []
    for mdl_path in mdl_to_scan:
        if not mdl_path.is_file():
            continue
        for rel in _texture_refs_in_mdl(mdl_path):
            dest = materials / rel
            if dest.is_file():
                continue
            source = _find_on_donors(rel, donors)
            if source is None:
                missing_textures.append(f"{mdl_path.name}:{rel}")
                continue
            _copy_file(source, dest)
            recovered_textures.append(rel)

    if recovered_mdl or recovered_textures:
        print(
            "[INFO] Recovered GRScenes materials into "
            f"{materials}: mdl={len(recovered_mdl)} "
            f"synthesized={len(synthesized_mdl)} "
            f"textures={len(recovered_textures)}",
            flush=True,
        )
        for name in recovered_mdl[:20]:
            kind = "synthesized" if name in synthesized_mdl else "copied"
            print(f"[INFO]   recovered mdl {name} ({kind})", flush=True)
        if len(recovered_mdl) > 20:
            print(f"[INFO]   ... {len(recovered_mdl) - 20} more mdl", flush=True)

    if still_missing and raise_if_missing:
        raise RuntimeError(
            "GRScenes materials are still missing after donor recovery; "
            "refusing to load a scene that would render error-red / "
            f"unresolved-gray. missing_mdl={still_missing} "
            f"collection={materials} donors={[str(d) for d in donors]}"
        )
    if missing_textures and raise_if_missing:
        raise RuntimeError(
            "GRScenes MDL textures are still missing after donor recovery; "
            "refusing to load a scene that would render unresolved materials. "
            f"missing_textures={missing_textures[:20]}"
            f"{' ...' if len(missing_textures) > 20 else ''}"
        )

    return {
        "recovered_mdl": recovered_mdl,
        "recovered_textures": recovered_textures,
        "referenced_mdl": referenced,
        "still_missing_mdl": still_missing,
        "synthesized_mdl": synthesized_mdl,
        "missing_textures": missing_textures,
    }


def _scene_layer_candidates(scene_path: Path) -> list[Path]:
    scene_dir = scene_path if scene_path.is_dir() else scene_path.parent
    layers = list(_iter_metadata_instance_usd(scene_path))
    for name in (
        "start_result_raw.usd",
        "start_result_raw.dedup.usd",
        "start_result_navigation.usd",
        "start_result_navigation.dedup.usd",
        "start_result_interaction.usd",
        "start_result_interaction.dedup.usd",
    ):
        candidate = scene_dir / name
        if candidate.is_file():
            layers.append(candidate)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in layers:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def rewrite_scene_material_asset_paths(scene_path: Path) -> int:
    """Rewrite leftover ``./Materials/Textures`` and HTTP MDL assets on disk.

    Crate layers intern-split the path, so a raw-bytes scan misses them.  Use
    Sdf.  ``info:mdl:sourceAsset`` stays ``Materials/Num….mdl`` (MDLC); remote
    vMaterials URLs become that same local form; texture inputs become
    layer-relative paths into the collection Materials tree.
    """
    materials = materials_dir_for(scene_path)
    edits = 0
    touched = 0
    for path in _scene_layer_candidates(scene_path):
        try:
            changed = rewrite_relative_mdl_source_assets_on_disk(
                path,
                materials_dir=materials,
            )
        except Exception as exc:
            print(f"[WARN] material asset rewrite skipped {path}: {exc}", flush=True)
            continue
        if changed:
            touched += 1
            edits += changed
    if edits:
        print(
            f"[INFO] Rewrote {edits} leftover ./Materials asset path(s) on "
            f"{touched} layer(s) before Kit open.",
            flush=True,
        )
    return edits


def configure_usd_asset_search_path(scene_path: Path) -> list[str]:
    """Put the collection / scene roots on USD's default asset search path."""
    roots: list[str] = []
    collection = collection_root_for(scene_path)
    scene_dir = scene_path if scene_path.is_dir() else scene_path.parent
    if collection is not None:
        roots.append(str(collection.resolve()))
    roots.append(str(scene_dir.resolve()))
    materials = materials_dir_for(scene_path)
    if materials is not None:
        roots.append(str(materials.resolve()))
    existing = [
        item
        for item in os.environ.get("PXR_AR_DEFAULT_SEARCH_PATH", "").split(os.pathsep)
        if item
    ]
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*roots, *existing]:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    os.environ["PXR_AR_DEFAULT_SEARCH_PATH"] = os.pathsep.join(merged)
    # Do not import pxr here. prepare() runs before SimulationApp; loading
    # conda usd-core first makes Kit segfault (Tf_PyEnumWrapper / rc=139).
    print(
        "[INFO] USD asset search path includes: " + ", ".join(roots),
        flush=True,
    )
    return merged


def _rewrite_material_assets_via_subprocess(scene_path: Path) -> int:
    """Run the Sdf rewrite in a child so prepare() never imports pxr."""
    cache_root = (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "tarsphere"
        / "material_rewrite"
    )
    cache_key = hashlib.sha256(str(scene_path.resolve()).encode("utf-8")).hexdigest()
    cache_path = cache_root / f"{cache_key}.json"

    def layer_signature() -> list[list[object]]:
        signature: list[list[object]] = []
        for path in _scene_layer_candidates(scene_path):
            try:
                stat = path.stat()
            except OSError:
                continue
            signature.append(
                [
                    str(path.resolve()),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                ]
            )
        return signature

    signature_before = layer_signature()
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if isinstance(cached, dict) and cached.get("layers") == signature_before:
        print(
            f"[INFO] Material asset rewrite cache hit for {scene_path.name} "
            f"({len(signature_before)} layers).",
            flush=True,
        )
        return 0

    env = os.environ.copy()
    repo = str(_REPO_ROOT)
    existing = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    env["PYTHONPATH"] = os.pathsep.join([repo, *existing])
    result = subprocess.run(
        [sys.executable, "-c", _REWRITE_CODE, str(scene_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    for line in (result.stdout or "").splitlines():
        print(line, flush=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError(
            "Material asset rewrite failed before Kit open "
            f"(rc={result.returncode}): {' | '.join(tail)}"
        )
    count = 0
    for line in (result.stdout or "").splitlines():
        if line.startswith("REWRITE_COUNT "):
            count = int(line.split()[-1])
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp-{os.getpid()}")
        temp_path.write_text(
            json.dumps(
                {
                    "schema_version": "tarsphere.material_rewrite_cache.v1",
                    "scene_path": str(scene_path.resolve()),
                    # A rewrite can change layer mtimes, so cache the post-pass
                    # state rather than the signature sampled before Sdf ran.
                    "layers": layer_signature(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temp_path.replace(cache_path)
    except OSError as exc:
        print(f"[WARN] Could not update material rewrite cache: {exc}", flush=True)
    return count


def prepare_grscene_materials(scene_path: Path) -> list[str]:
    """Recover missing modules and set search paths. Call before Kit.

    Texture-asset and HTTP-MDL rewrites use Sdf and must run in a *separate*
    process (or after SimulationApp owns pxr).  ``rewrite_scene_material_asset_paths``
    is that helper; BEV finalize still rewrites leftover paths in-memory.
    """
    recover_missing_collection_materials(scene_path)
    _rewrite_material_assets_via_subprocess(scene_path)
    configure_usd_asset_search_path(scene_path)
    return configure_mdl_system_path(scene_path)


def ensure_material_deduped_usd(scene_usd: Path) -> Path:
    """Load ``<scene>.dedup.usd``, baking on demand. Shared by BEV and runtime.

    The asset preparation pipeline writes a root stage that sublayers the
    untouched original. Episodes and canonical BEVs both resolve the evaluation
    navigation source through this function. Explicit custom episode files keep
    their own overlay; a raw sibling never substitutes for a navigation source.
    """
    value = os.environ.get("TARSPHERE_SCENE_MATERIAL_DEDUP", "").strip().lower()
    if value in {"0", "false", "no", "off"}:
        print(
            f"[INFO] TARSPHERE_SCENE_MATERIAL_DEDUP={value}: loading the scene as shipped "
            "(no material dedup; expect a much slower load).",
            flush=True,
        )
        return scene_usd

    if scene_usd.name.endswith(".dedup.usd"):
        return scene_usd

    deduped = scene_usd.with_suffix(f".dedup{scene_usd.suffix}")
    from tarsphere.infra.simulation.material_dedup import overlay_is_current

    stale = deduped.is_file() and not overlay_is_current(scene_usd, deduped)
    if stale:
        print(
            f"[INFO] {deduped.name} has an outdated generator or changed dependencies; re-baking.",
            flush=True,
        )
    if not deduped.is_file() or stale:
        if not os.access(scene_usd.parent, os.W_OK):
            print(
                f"[WARN] Cannot cache a material dedup bake in {scene_usd.parent} "
                "(not writable); loading the original scene. Install the "
                f"pre-baked material stage for {scene_usd} or set "
                "TARSPHERE_SCENE_MATERIAL_DEDUP=0 to silence this.",
                flush=True,
            )
            return scene_usd
        if not _bake_material_deduped_scene(scene_usd, deduped):
            return scene_usd
    print(f"[INFO] Loading material-deduplicated scene {deduped.name}.", flush=True)
    return deduped


def _bake_material_deduped_scene(scene_usd: Path, deduped: Path) -> bool:
    print(
        f"[INFO] Baking material-deduplicated scene {deduped.name} (one-off, "
        "reused while scene and material dependencies are unchanged).",
        flush=True,
    )
    started = time.monotonic()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(_REPO_ROOT),
            *filter(None, environment.get("PYTHONPATH", "").split(os.pathsep)),
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tarsphere.infra.simulation.material_dedup",
            str(scene_usd),
            "--output",
            str(deduped),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    for line in (result.stdout or "").splitlines():
        print(f"[BAKE] {line}", flush=True)
    if result.returncode != 0 or not deduped.is_file():
        tail = (result.stderr or "").strip().splitlines()[-3:]
        print(
            f"[WARN] Material dedup bake failed (rc={result.returncode}); "
            f"loading the original scene. {' | '.join(tail)}",
            flush=True,
        )
        return False
    print(
        f"[INFO] Baked {deduped.name} in {time.monotonic() - started:.1f}s.",
        flush=True,
    )
    return True


def resolve_runtime_scene_usd(scene_path: Path) -> Path:
    """Episode / Isaac contract: directory → ``start_result_navigation.usd`` + dedup."""
    if scene_path.is_dir():
        scene_usd = scene_path / "start_result_navigation.usd"
        if not scene_usd.is_file():
            raise FileNotFoundError(
                f"Scene directory does not contain start_result_navigation.usd: {scene_path}"
            )
    else:
        scene_usd = scene_path
    return ensure_material_deduped_usd(scene_usd)


def rewrite_relative_mdl_source_assets_on_stage(
    stage: object,
    *,
    scene_path: Path | None = None,
) -> int:
    """Rewrite leftover ``./Materials`` MDL and texture assets (in-memory)."""
    layers = getattr(stage, "GetUsedLayers", None)
    if not callable(layers):
        return 0
    materials = materials_dir_for(scene_path) if scene_path is not None else None
    changed = 0
    for layer in layers():
        changed += rewrite_relative_mdl_source_assets_in_layer(
            layer,
            materials_dir=materials,
        )
    if changed:
        print(
            f"[INFO] Rewrote {changed} leftover ./Materials asset path(s) "
            "on the opened stage.",
            flush=True,
        )
    return changed


def missing_mdl_on_stage(stage: object, materials: Path | None) -> list[str]:
    """Basenames of ``info:mdl:sourceAsset`` that do not exist on disk."""
    if materials is None or not materials.is_dir():
        return []
    present = {path.name for path in materials.glob("*.mdl")}
    for root in resolve_kit_stock_mdl_search_paths():
        present.update(path.name for path in root.glob("*.mdl"))
    missing: set[str] = set()
    layers = getattr(stage, "GetUsedLayers", None)
    if not callable(layers):
        return []

    def visit(prim_spec: object) -> None:
        attributes = getattr(prim_spec, "attributes", None)
        if attributes is not None:
            attr = attributes.get("info:mdl:sourceAsset")
            if attr is not None and attr.default is not None:
                current = _asset_path_text(attr.default)
                basename = Path(str(current)).name
                if basename.endswith(".mdl") and basename not in present:
                    missing.add(basename)
        for child in getattr(prim_spec, "nameChildren", ()):
            visit(child)

    for layer in layers():
        root = getattr(layer, "pseudoRoot", None)
        if root is not None:
            visit(root)
    return sorted(missing)


def _kit_update(update_fn: Callable[[], object] | None) -> None:
    if update_fn is not None:
        update_fn()
        return
    import omni.kit.app

    omni.kit.app.get_app().update()


def _stage_loading_status() -> tuple[bool, int, int]:
    import omni.usd

    is_loading, files_loaded, total_files = (
        omni.usd.get_context().get_stage_loading_status()
    )
    return bool(is_loading), int(files_loaded), int(total_files)


def dump_unresolved_material_assets(label: str = "materials") -> list[str]:
    """Print remote / missing MDL sourceAssets still on the opened stage.

    Used when the working set stalls so the leftover files are named instead
    of only ``loaded=N/N+9``.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    found: list[str] = []
    seen: set[str] = set()
    if stage is None:
        print(f"[{label}] wait dump: no USD stage is open.", flush=True)
        return found

    def visit(prim_spec: object) -> None:
        attributes = getattr(prim_spec, "attributes", None)
        if attributes is not None:
            for name, attr in attributes.items():
                if attr is None or attr.default is None:
                    continue
                current = _asset_path_text(attr.default)
                if not current:
                    continue
                if name != "info:mdl:sourceAsset" and not is_remote_material_asset(
                    current
                ):
                    continue
                key = f"{name}={current}"
                if key in seen:
                    continue
                seen.add(key)
                found.append(key)
        for child in getattr(prim_spec, "nameChildren", ()):
            visit(child)

    layers = getattr(stage, "GetUsedLayers", None)
    if callable(layers):
        for layer in layers():
            root = getattr(layer, "pseudoRoot", None)
            if root is not None:
                visit(root)
    remotes = [
        item for item in found if is_remote_material_asset(item.split("=", 1)[-1])
    ]
    print(
        f"[{label}] wait dump: {len(remotes)} remote material asset(s), "
        f"{len(found)} named sourceAsset(s).",
        flush=True,
    )
    for item in remotes[:20]:
        print(f"[{label}] wait dump remote: {item}", flush=True)
    if len(remotes) > 20:
        print(f"[{label}] wait dump remote: ... {len(remotes) - 20} more", flush=True)
    return remotes


def _available_memory_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            kib = int(line.split()[1])
            return float(kib) / (1024.0 * 1024.0)
    return None


def wait_for_material_working_set(
    update_fn: Callable[[], object] | None = None,
    *,
    label: str = "materials",
    require_idle: bool = True,
    max_updates: int | None = None,
    max_sec: float | None = None,
    confirmation_frames: int = _DEFAULT_CONFIRMATION_FRAMES,
    min_available_memory_gib: float = _DEFAULT_MIN_AVAILABLE_GIB,
    allow_verified_stable_plateau: bool = False,
    stable_plateau_frames: int = 600,
    unseen_idle_warmup_frames: int = 90,
) -> dict[str, object]:
    """Block until Kit has finished streaming the material working set.

    Ready means Kit explicitly reports that the loader is inactive and its
    counters are idle (0 / 0), or that it has reached 100% of the current
    total. An idle 0 / 0 is also the valid result for a scene with no material
    work or a fully cached working set; requiring a non-zero counter first
    makes those scenes time out. A 50% threshold is not enough: AEI8 BEV
    captured at 81% with 1741 files still pending and lost shelf / sofa
    materials. This wait is used by BEV *and* episode runtime.
    """
    max_updates = (
        int(
            _env_flag(
                "TARSPHERE_MATERIAL_WAIT_MAX_UPDATES",
                default=str(_DEFAULT_WAIT_MAX_UPDATES),
            )
        )
        if max_updates is None
        else int(max_updates)
    )
    max_sec = (
        float(
            _env_flag(
                "TARSPHERE_MATERIAL_WAIT_MAX_SEC", default=str(_DEFAULT_WAIT_MAX_SEC)
            )
        )
        if max_sec is None
        else float(max_sec)
    )
    # Default strict=True.  Only TARSPHERE_MATERIAL_WAIT_STRICT=0 relaxes a timeout.
    strict = _env_flag(
        "TARSPHERE_MATERIAL_WAIT_STRICT", default="1"
    ).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    consecutive_ready = 0
    loading_observed = False
    peak_total_files = 0
    minimum_available = float("inf")
    deadline = time.monotonic() + max_sec
    last_loaded = 0
    last_total = 0
    last_fraction = 0.0
    last_is_loading = False
    last_status: tuple[int, int] | None = None
    unchanged_frames = 0

    for frame_idx in range(max(1, max_updates)):
        _kit_update(update_fn)
        available = _available_memory_gib()
        if available is not None:
            minimum_available = min(minimum_available, available)
            if available < min_available_memory_gib:
                raise RuntimeError(
                    f"{label} material streaming stopped before system OOM: "
                    f"available memory fell to {available:.2f} GiB."
                )
        is_loading, files_loaded, total_files = _stage_loading_status()
        last_is_loading = is_loading
        last_loaded, last_total = files_loaded, total_files
        status = (files_loaded, total_files)
        if status == last_status:
            unchanged_frames += 1
        else:
            unchanged_frames = 0
            last_status = status
        loading_observed = (
            loading_observed or is_loading or bool(files_loaded or total_files)
        )
        peak_total_files = max(peak_total_files, total_files)
        last_fraction = (
            float(files_loaded) / float(total_files) if total_files > 0 else 0.0
        )
        if frame_idx == 0 or (frame_idx + 1) % 50 == 0:
            print(
                f"[{label}] RGB material streaming: "
                f"frame={frame_idx + 1}, files_loaded={files_loaded}, "
                f"total_files={total_files}, is_loading={is_loading}",
                flush=True,
            )
        idle = not is_loading and files_loaded == 0 and total_files == 0
        if (
            idle
            and not loading_observed
            and (frame_idx + 1) < max(0, int(unseen_idle_warmup_frames))
        ):
            # Immediate 0/0 is "loader has not started", not a cached working set.
            # Accept that idle only after a warmup or after Kit has shown work.
            idle = False
        complete = (
            not is_loading
            and total_files > 0
            and files_loaded >= total_files
            and (not require_idle or files_loaded == total_files)
        )
        ready = idle or complete
        verified_plateau = False
        if (
            not ready
            and allow_verified_stable_plateau
            and files_loaded > 0
            and total_files > files_loaded
            and unchanged_frames >= max(1, int(stable_plateau_frames))
        ):
            # Isaac's optional Material Library browser indexes every module
            # on MDL_SYSTEM_PATH and contaminates the global stage-loading
            # counter even though those catalog entries are not referenced by
            # the opened stage.  For one-shot BEV rendering only, a stable
            # plateau is acceptable after the pre-open local MDL/texture
            # inventory passed and the composed stage has no remote assets.
            # Runtime callers retain the strict idle/100% contract.
            verified_plateau = not dump_unresolved_material_assets(label)
            ready = verified_plateau
        if not ready:
            consecutive_ready = 0
            if time.monotonic() >= deadline:
                break
            continue
        consecutive_ready += 1
        if consecutive_ready >= confirmation_frames:
            print(
                f"[{label}] RGB material streaming: working set ready after "
                f"{frame_idx + 1} frame(s), loaded_fraction="
                f"{last_fraction:.4f} idle={idle}",
                flush=True,
            )
            return _material_wait_telemetry(
                updates=frame_idx + 1,
                confirmation_frames=consecutive_ready,
                files_loaded=files_loaded,
                total_files=total_files,
                last_fraction=last_fraction,
                peak_total_files=peak_total_files,
                loading_observed=loading_observed,
                minimum_available=minimum_available,
                idle=idle,
                verified_plateau=verified_plateau,
                incomplete=False,
            )
        if time.monotonic() >= deadline:
            break

    leftover = dump_unresolved_material_assets(label)
    leftover_note = ""
    if leftover:
        leftover_note = f" remote_assets={leftover[:9]}"
    message = (
        f"{label} material streaming did not finish "
        f"(loaded={last_loaded}/{last_total} fraction={last_fraction:.4f} "
        f"peak_total={peak_total_files} is_loading={last_is_loading} "
        f"observed={loading_observed}"
        f"{leftover_note})."
    )
    print(f"[ERROR] {message}", flush=True)
    if strict:
        raise RuntimeError(message)
    print(
        f"[WARN] {message} Continuing because TARSPHERE_MATERIAL_WAIT_STRICT=0.",
        flush=True,
    )
    return _material_wait_telemetry(
        updates=max_updates,
        confirmation_frames=consecutive_ready,
        files_loaded=last_loaded,
        total_files=last_total,
        last_fraction=last_fraction,
        peak_total_files=peak_total_files,
        loading_observed=loading_observed,
        minimum_available=minimum_available,
        idle=False,
        verified_plateau=False,
        incomplete=True,
    )


def _material_wait_telemetry(
    *,
    updates: int,
    confirmation_frames: int,
    files_loaded: int,
    total_files: int,
    last_fraction: float,
    peak_total_files: int,
    loading_observed: bool,
    minimum_available: float,
    idle: bool,
    verified_plateau: bool,
    incomplete: bool,
) -> dict[str, object]:
    """Return the one complete material-wait telemetry record.

    ``rgb_material_wait_idle`` is always a bool.  A verified catalog plateau
    is a ready working set even when Kit's global counters never return to
    0/0; lighting classification must read ``rgb_material_working_set_ready``,
    not a 50% leftover fraction from an older wait target.
    """

    return {
        "rgb_material_wait_updates": int(updates),
        "rgb_material_ready_confirmation_frames": int(confirmation_frames),
        "rgb_material_loaded_files_at_capture": int(files_loaded),
        "rgb_material_total_files_at_capture": int(total_files),
        "rgb_material_loaded_fraction_at_capture": (
            1.0 if idle and not total_files else round(float(last_fraction), 6)
        ),
        "rgb_material_target_loaded_fraction": 1.0,
        "rgb_material_peak_total_files": int(peak_total_files),
        "rgb_material_loading_observed": bool(loading_observed),
        "rgb_material_min_available_memory_gib": (
            round(minimum_available, 3) if minimum_available < float("inf") else None
        ),
        "rgb_material_wait_idle": bool(idle),
        "rgb_material_wait_verified_stable_plateau": bool(verified_plateau),
        "rgb_material_working_set_ready": bool(
            (idle or verified_plateau) and not incomplete
        ),
        "rgb_material_wait_incomplete": bool(incomplete),
    }


def finalize_loaded_scene_materials(
    scene_path: Path,
    update_fn: Callable[[], object] | None = None,
    *,
    label: str = "runtime",
    allow_verified_stable_plateau: bool = False,
) -> dict[str, object]:
    """Rewrite leftover ``./`` refs, then wait for a complete working set."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Cannot finalize scene materials: no USD stage is open.")
    print(
        f"[{label}] finalize materials: rewrite leftover ./ and HTTP refs...",
        flush=True,
    )
    rewrite_relative_mdl_source_assets_on_stage(stage, scene_path=scene_path)
    materials = materials_dir_for(scene_path)
    print(f"[{label}] finalize materials: check missing MDL...", flush=True)
    missing = missing_mdl_on_stage(stage, materials)
    if missing:
        # One last recover from donors using the composed-stage inventory, then
        # fail if anything is still absent.  Do not remap to WorldGrid — that
        # is the plastic-white / gray look the user rejected.  Pass the stage
        # names explicitly: crate intern-split hides named vMaterials from the
        # raw-bytes scan that recover uses before Kit starts.
        recover_missing_collection_materials(
            scene_path,
            extra_referenced=missing,
        )
        missing = missing_mdl_on_stage(stage, materials)
        if missing:
            raise RuntimeError(
                "Opened stage still references MDL modules that do not exist "
                f"under {materials}: {missing}. Recover them from the home "
                "Materials twin, the official GRScenes commercial zip, or "
                "Kit/vMaterials stock before loading this scene."
            )
    print(f"[{label}] finalize materials: wait working set...", flush=True)
    wait_meta = wait_for_material_working_set(
        update_fn,
        label=label,
        allow_verified_stable_plateau=allow_verified_stable_plateau,
        max_updates=int(
            _env_flag("TARSPHERE_MATERIAL_WAIT_MAX_UPDATES", default="20000")
        ),
        max_sec=float(_env_flag("TARSPHERE_MATERIAL_WAIT_MAX_SEC", default="1800")),
    )
    return wait_meta


__all__ = [
    "collection_root_for",
    "configure_usd_asset_search_path",
    "ensure_material_deduped_usd",
    "ensure_stock_named_modules",
    "finalize_loaded_scene_materials",
    "materials_dir_for",
    "prepare_grscene_materials",
    "recover_missing_collection_materials",
    "referenced_mdl_basenames",
    "resolve_runtime_scene_usd",
    "stock_material_donors",
    "dump_unresolved_material_assets",
    "rewrite_relative_mdl_source_assets_on_stage",
    "rewrite_scene_material_asset_paths",
    "sibling_material_donors",
    "wait_for_material_working_set",
]
