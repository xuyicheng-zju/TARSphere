"""Stub unused native sensor extensions before Kit starts.

The shipped physics-sensor plugin can crash during sustained contact-report
traffic, and disabling its extension does not unload the native plugin.
TARSphere monitors contacts through PhysX directly. Higher-version stubs keep
Kit's sensor dependencies importable without loading that plugin.

Set ``TARSPHERE_ALLOW_SENSORS_PHYSICS_EXT=1`` before startup to use the shipped
Contact, IMU, and Effort sensor extension.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STUB_EXTENSIONS_DIR = _REPO_ROOT / "extensions"
_TARSPHERE_ISAAC_KIT = (
    Path(__file__).resolve().parents[1] / "kit" / "tarsphere_isaac_python.kit"
)
_PATCHED = False


def _allow_shipped_sensors_physics_ext() -> bool:
    value = os.environ.get("TARSPHERE_ALLOW_SENSORS_PHYSICS_EXT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _force_sync_material_loads() -> bool:
    """Read the opt-in setting for synchronous MDL material loading.

    Asynchronous loading avoids blocking a scene update on all materials.
    Set ``TARSPHERE_SYNC_MATERIAL_LOADS=1`` before startup to use Isaac's
    synchronous behavior for render diagnostics.
    """
    value = os.environ.get("TARSPHERE_SYNC_MATERIAL_LOADS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _kit_active_gpu() -> int | None:
    """Resolve the physical GPU used by Kit rendering and physics.

    Explicit demo isolation supplies a verified logical-to-physical mapping.
    Other launches may set ``TARSPHERE_KIT_ACTIVE_GPU``; when unset, Kit uses
    its default device selection.
    """
    from tarsphere.infra.simulation.demo_gpu import resolve_demo_gpu_mapping

    mapping = resolve_demo_gpu_mapping()
    if mapping is not None:
        return mapping["physical_gpu"]
    raw = os.environ.get("TARSPHERE_KIT_ACTIVE_GPU", "").strip()
    if not raw:
        return None
    try:
        gpu = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"TARSPHERE_KIT_ACTIVE_GPU must be an integer CUDA index, got {raw!r}"
        ) from exc
    if gpu < 0:
        raise RuntimeError(f"TARSPHERE_KIT_ACTIVE_GPU must be >= 0, got {gpu}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != "":
        raise RuntimeError(
            "Isaac must leave CUDA_VISIBLE_DEVICES unset so TARSPHERE_KIT_ACTIVE_GPU "
            f"is a physical index (got CUDA_VISIBLE_DEVICES={visible!r}, "
            f"TARSPHERE_KIT_ACTIVE_GPU={gpu}). Model services own GPU0; Kit uses GPU1."
        )
    return gpu


def _apply_kit_active_gpu(config: dict) -> None:
    from tarsphere.infra.simulation.demo_gpu import resolve_demo_gpu_mapping

    mapping = resolve_demo_gpu_mapping()
    if mapping is not None:
        config["active_gpu"] = mapping["physical_gpu"]
        config["physics_gpu"] = mapping["cuda_ordinal"]
        config["multi_gpu"] = False
        print(
            "[INFO] Explicit demo GPU isolation: "
            f"renderer/activeGpu={mapping['physical_gpu']}, "
            f"physics/cudaDevice={mapping['cuda_ordinal']}, "
            f"CUDA_VISIBLE_DEVICES={mapping['uuid']} (NVML identity verified).",
            flush=True,
        )
        return
    gpu = _kit_active_gpu()
    if gpu is None:
        return
    config["active_gpu"] = gpu
    config["physics_gpu"] = gpu
    config["multi_gpu"] = False
    print(
        f"[INFO] TARSPHERE_KIT_ACTIVE_GPU={gpu}: pinning Kit "
        f"renderer/activeGpu={gpu} and physics/cudaDevice={gpu} "
        "(CUDA_VISIBLE_DEVICES unset, multi_gpu=False).",
        flush=True,
    )


def kit_runtime_extra_args() -> list[str]:
    """Give each locked runtime slot local Kit state.

    XDG/OMNI cache variables do not isolate Kit's portable data directory.
    Do not inherit another process's persistent render settings or share its
    derived-data locks. Reset-user ignores the settings; it deletes no files.
    """
    from tarsphere.infra.simulation.isaac_process_lock import isaac_runtime_slot

    directory = (
        Path(tempfile.gettempdir())
        / f"tarsphere_kit_{os.getuid()}"
        / isaac_runtime_slot()
    )
    directory.mkdir(parents=True, exist_ok=True)
    return ["--portable-root", str(directory), "--reset-user"]


def ensure_sensors_physics_stub() -> None:
    """Patch TARSphere's SimulationApp bootstrap once per process."""
    global _PATCHED
    if _PATCHED or _allow_shipped_sensors_physics_ext():
        return

    stub_toml = (
        _STUB_EXTENSIONS_DIR / "isaacsim.sensors.physics" / "config" / "extension.toml"
    )
    if not stub_toml.is_file():
        raise RuntimeError(f"Missing sensors.physics stub extension: {stub_toml}")
    compatibility_stub_toml = (
        _STUB_EXTENSIONS_DIR / "omni.isaac.sensor" / "config" / "extension.toml"
    )
    if not compatibility_stub_toml.is_file():
        raise RuntimeError(
            "Missing deprecated sensor compatibility stub extension: "
            f"{compatibility_stub_toml}"
        )
    if not _TARSPHERE_ISAAC_KIT.is_file():
        raise RuntimeError(
            f"Missing TARSphere Isaac kit experience: {_TARSPHERE_ISAAC_KIT}"
        )

    import isaacsim

    original_cls = isaacsim.SimulationApp
    stub_folder = str(_STUB_EXTENSIONS_DIR.resolve())
    experience = str(_TARSPHERE_ISAAC_KIT.resolve())
    isaac_root = Path(os.environ["ISAAC_SIM_PATH"])
    # The custom TARSphere experience lives outside Isaac's apps/ directory, so it
    # cannot inherit the default experience's sibling extension folders.
    extension_folders = [
        isaac_root / name for name in ("extscache", "extsUser", "extsDeprecated")
    ]
    extension_args: list[str] = []
    for folder in extension_folders:
        if not folder.is_dir():
            raise RuntimeError(f"Isaac extension folder is missing: {folder}")
        extension_args.extend(("--ext-folder", str(folder.resolve())))

    def patched_simulation_app(launch_config=None, *args, **kwargs):
        # SimulationApp.__init__ applies ``launch_config`` after it resolves the
        # default experience, so these keys win over Isaac's own choice.
        config = dict(launch_config or {})
        config["experience"] = experience
        _apply_kit_active_gpu(config)
        from tarsphere.infra.simulation.mdl_search_paths import (
            additional_system_path_extra_args,
        )

        config["extra_args"] = [
            *(config.get("extra_args") or []),
            *kit_runtime_extra_args(),
            "--ext-folder",
            stub_folder,
            *extension_args,
            *additional_system_path_extra_args(),
        ]
        if _force_sync_material_loads():
            print(
                "[INFO] TARSPHERE_SYNC_MATERIAL_LOADS=1: keeping Isaac synchronous MDL "
                "material loading (expect the multi-minute first-update stall).",
                flush=True,
            )
        else:
            # SimulationApp derives --/rtx/materialDb/syncLoads=True,
            # --/rtx/hydra/materialSyncLoads=True and
            # --/omni.kit.plugin/syncUsdLoads=True from its ``sync_loads``
            # launch-config key.  Override only the two material keys, appended
            # after Isaac's own args so the last assignment wins; leave
            # syncUsdLoads alone so a stage is still fully composed when
            # open_stage returns (sync_loads=False would relax that too).
            config["extra_args"] += [
                "--/rtx/materialDb/syncLoads=false",
                "--/rtx/hydra/materialSyncLoads=false",
            ]
            print(
                "[INFO] Loading MDL materials asynchronously "
                "(rtx/materialDb/syncLoads=false, rtx/hydra/materialSyncLoads="
                "false). Set TARSPHERE_SYNC_MATERIAL_LOADS=1 to restore Isaac's "
                "synchronous loading.",
                flush=True,
            )
        if os.environ.get("TARSPHERE_VERBOSE_NAVMESH", "").strip() in (
            "1",
            "true",
            "True",
        ):
            # ``omni.anim.navigation.core`` reports why it refuses to bake
            # ("unable to start baking because the stage does not contain
            # compatible geometry that intersects with at least one NavMesh
            # volume", and two sibling messages) on its own log channel at info
            # level, which Kit drops by default.
            config["extra_args"] += [
                "--/log/level=info",
                "--/log/channels/omni.anim.navigation.core=info",
            ]
            print(
                "[INFO] TARSPHERE_VERBOSE_NAVMESH=1: raising the "
                "omni.anim.navigation.core log channel to info.",
                flush=True,
            )
        print(
            "[INFO] Registered TARSphere sensor stubs "
            f"(experience={experience}, ext-folder={stub_folder}). "
            "Set TARSPHERE_ALLOW_SENSORS_PHYSICS_EXT=1 to use the shipped extension.",
            flush=True,
        )
        return original_cls(config, *args, **kwargs)

    # Wrap the app constructor rather than TARSphere's setup_isaacsim: that
    # method also owns streaming setup, and copying its body here would fork
    # silently the next time TARSphere changes it.
    isaacsim.SimulationApp = patched_simulation_app
    _PATCHED = True
