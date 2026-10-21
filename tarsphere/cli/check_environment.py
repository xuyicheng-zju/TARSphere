#!/usr/bin/env python
"""Fail-loud preflight for the current TARSphere headless runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from tarsphere.cli.runtime import runtime_issues

SINGLE_SERVER_CAPACITY = (8, 32)
MINIMUM_GPU_MEMORY_MIB = 23_000
MINIMUM_FREE_STORAGE_GIB = 50


@dataclass(frozen=True)
class Verification:
    """One explicit deployment condition and its observed value."""

    description: str
    passed: bool
    observed: str
    required: bool = True


def _gpus() -> list[tuple[str, int, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    gpus: list[tuple[str, int, str]] = []
    from tarsphere.cli.gpu_selection import select_gpu_indices

    rows = [
        [part.strip() for part in line.split(",")]
        for line in result.stdout.splitlines()
    ]
    if any(len(parts) != 5 for parts in rows):
        return []
    permitted = select_gpu_indices(
        [(int(parts[0]), parts[1]) for parts in rows],
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    for parts in rows:
        if int(parts[0]) in permitted:
            gpus.append((parts[2], int(float(parts[3])), parts[4]))
    return gpus


def _physical_cpu_cores() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        sockets: set[tuple[str, str]] = set()
        for block in cpuinfo.read_text(encoding="utf-8").split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            if "physical id" in fields and "core id" in fields:
                sockets.add((fields["physical id"], fields["core id"]))
        if sockets:
            return len(sockets)
    return os.cpu_count() or 0


def _system_memory_gib() -> float:
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    return page_size * physical_pages / (1024**3)


def _pinned_scene_assets(
    asset_root: Path,
    episodes: list[Any],
    *,
    require_visual_maps: bool = False,
) -> tuple[int, list[str]]:
    """Check the actual scene and maps declared by each selected episode."""
    scene_paths: set[Path] = set()
    required_paths: set[Path] = set()
    missing: list[str] = []

    def asset_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else asset_root / path

    for episode in episodes:
        scene = asset_path(episode.scene.asset_path)
        scene_paths.add(scene)
        required_paths.add(scene)
        scene_map = (episode.world_knowledge or {}).get("scene_map")
        if not isinstance(scene_map, dict):
            missing.append(
                f"{episode.episode_id}: world_knowledge.scene_map is missing"
            )
            continue
        required_map_fields = (
            ("occupancy", "rgb_bev", "projection")
            if require_visual_maps
            else ("occupancy",)
        )
        for field in required_map_fields:
            value = scene_map.get(field)
            if not isinstance(value, str) or not value.strip():
                missing.append(f"{episode.episode_id}: scene_map.{field} is missing")
        for field in required_map_fields:
            value = scene_map.get(field)
            if isinstance(value, str) and value.strip():
                required_paths.add(asset_path(value))
    missing.extend(str(path) for path in sorted(required_paths) if not path.is_file())
    return len(scene_paths), missing


def _selected_runtime_assets(asset_root: Path, episodes: list[Any]) -> list[Path]:
    """Resolve the robot assets that the selected episodes actually instantiate."""
    from tarsphere.embodiment.robot_profiles import get_robot_profile

    paths = {
        asset_root / get_robot_profile(episode.robot.robot_type).usd_rel_path
        for episode in episodes
    }
    if any(episode.humans for episode in episodes):
        paths.add(
            asset_root / "characters" / "Animations" / "stand_walk_loop.skelanim.usd"
        )
    return sorted(paths)


def _formal_episode_asset_payloads(
    asset_root: Path,
    episodes: list[Any],
) -> tuple[int, list[str]]:
    """Resolve every injected object payload used by the supplied episodes."""
    referenced = {
        instance["asset_path"]
        for episode in episodes
        for instance in (
            (episode.world_knowledge or {})
            .get("asset_adaptation", {})
            .get("instances", [])
        )
    }
    missing = [
        str(asset_root / relative_path)
        for relative_path in sorted(referenced)
        if not (asset_root / relative_path).is_file()
    ]
    return len(referenced), missing


def _formal_episode_material_dependencies(
    asset_root: Path,
    episodes: list[Any],
) -> tuple[int, list[str]]:
    """Compose formal payloads and require every authored material asset."""
    from pxr import Sdf, Usd

    from tarsphere.infra.simulation.scene_materials import materials_dir_for

    referenced = {
        instance["asset_path"]
        for episode in episodes
        for instance in (
            (episode.world_knowledge or {})
            .get("asset_adaptation", {})
            .get("instances", [])
        )
    }
    dependencies: set[Path] = set()
    missing: list[str] = []
    for relative_path in sorted(referenced):
        payload_path = asset_root / relative_path
        if not payload_path.is_file():
            missing.append(f"{payload_path}: missing payload")
            continue
        stage = Usd.Stage.Open(str(payload_path))
        if stage is None:
            missing.append(f"{payload_path}: cannot compose payload")
            continue
        material_root = materials_dir_for(payload_path)
        for prim in stage.Traverse():
            for attribute in prim.GetAttributes():
                value = attribute.Get()
                if not isinstance(value, Sdf.AssetPath) or not value.path:
                    continue
                authored_path = value.path.replace("\\", "/")
                marker = "Materials/"
                if marker not in authored_path:
                    continue
                resolved = Path(value.resolvedPath) if value.resolvedPath else None
                if resolved is None or not resolved.is_file():
                    resolved = (
                        material_root / authored_path.split(marker, 1)[1]
                        if material_root is not None
                        else None
                    )
                if resolved is None or not resolved.is_file():
                    missing.append(f"{payload_path}: {authored_path}")
                    continue
                dependencies.add(resolved.resolve())
    return len(dependencies), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    from tarsphere.cli.episode_inputs import DEFAULT_EPISODES, load_episode_inputs

    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Also validate continuous-video BEV map assets",
    )
    args = parser.parse_args()
    minimum_cores, minimum_memory = SINGLE_SERVER_CAPACITY
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    isaac_root = Path(os.environ.get("ISAAC_SIM_PATH", "")).expanduser()
    asset_root = Path(
        os.environ.get("TARSPHERE_ASSETS_PATH", REPOSITORY_ROOT / "assets")
    ).expanduser()

    checks: list[Verification] = []
    checks.append(
        Verification(
            "Server-side tmux is available",
            bool(shutil.which("tmux")),
            shutil.which("tmux") or "missing; install tmux before formal launch",
        )
    )
    checks.append(
        Verification(
            "Python is 3.11",
            sys.version_info[:2] == (3, 11),
            sys.version.split()[0],
        )
    )
    grasp_python = Path(os.environ.get("TARSPHERE_GRASPGENX_PYTHON", "")).expanduser()
    checks.append(
        Verification(
            "TARSPHERE_GRASPGENX_PYTHON is executable",
            grasp_python.is_file() and os.access(grasp_python, os.X_OK),
            str(grasp_python) if str(grasp_python) != "." else "unset",
        )
    )

    gpus = _gpus()
    checks.append(
        Verification(
            "At least one NVIDIA GPU is visible",
            len(gpus) >= 1,
            f"{len(gpus)} GPU(s)",
        )
    )
    if gpus:
        checks.append(
            Verification(
                "Every GPU is RTX-class and exposes hardware ray tracing",
                all("RTX" in gpu[0].upper() for gpu in gpus),
                ", ".join(gpu[0] for gpu in gpus),
            )
        )
        checks.append(
            Verification(
                "Every GPU has at least 23 GiB for the default runtime",
                all(gpu[1] >= MINIMUM_GPU_MEMORY_MIB for gpu in gpus),
                ", ".join(f"{gpu[1]} MiB" for gpu in gpus),
            )
        )

    physical_cores = _physical_cpu_cores()
    checks.append(
        Verification(
            f"At least {minimum_cores} physical CPU cores are available",
            physical_cores >= minimum_cores,
            f"{physical_cores} physical core(s)",
            required=False,
        )
    )
    system_memory_gib = _system_memory_gib()
    checks.append(
        Verification(
            f"At least {minimum_memory} GiB system memory is available",
            system_memory_gib >= minimum_memory,
            f"{system_memory_gib:.1f} GiB",
            required=False,
        )
    )

    isaac_version_path = isaac_root / "VERSION"
    isaac_version = (
        isaac_version_path.read_text(encoding="utf-8").strip()
        if isaac_version_path.is_file()
        else "missing"
    )
    checks.append(
        Verification(
            "ISAAC_SIM_PATH contains Isaac Sim 5.1 standalone",
            isaac_version.startswith("5.1.")
            and (isaac_root / "setup_conda_env.sh").is_file(),
            f"{isaac_root} ({isaac_version})",
        )
    )

    simulator_issues = runtime_issues()
    checks.append(
        Verification(
            "Built-in TARSphere simulation core and extensions are complete",
            not simulator_issues,
            "; ".join(simulator_issues) if simulator_issues else "tarsphere/simulator",
        )
    )

    formal_episodes = [
        episode for _, episode in load_episode_inputs(args.episodes, args.episode_id)
    ]
    expected_scene_count = len(
        {episode.scene.asset_path for episode in formal_episodes}
    )
    scene_count, missing_scene_assets = _pinned_scene_assets(
        asset_root,
        formal_episodes,
        require_visual_maps=args.record_video,
    )
    checks.append(
        Verification(
            f"All {expected_scene_count} declared scenes and scene maps are present",
            scene_count == expected_scene_count and not missing_scene_assets,
            (
                f"{scene_count}/{expected_scene_count} scenes complete"
                if not missing_scene_assets
                else f"{len(missing_scene_assets)} files missing; first: {missing_scene_assets[0]}"
            ),
        )
    )

    payload_count, missing_episode_payloads = _formal_episode_asset_payloads(
        asset_root,
        formal_episodes,
    )
    checks.append(
        Verification(
            "Every formal episode adaptation payload is present",
            not missing_episode_payloads,
            (
                f"{payload_count} unique payloads present"
                if not missing_episode_payloads
                else (
                    f"{len(missing_episode_payloads)}/{payload_count} payloads missing; "
                    f"first: {missing_episode_payloads[0]}"
                )
            ),
        )
    )
    dependency_count, missing_material_dependencies = (
        _formal_episode_material_dependencies(asset_root, formal_episodes)
    )
    checks.append(
        Verification(
            "Every formal episode payload material dependency is present",
            not missing_material_dependencies,
            (
                f"{dependency_count} unique material dependencies present"
                if not missing_material_dependencies
                else (
                    f"{len(missing_material_dependencies)} material dependencies "
                    f"missing; first: {missing_material_dependencies[0]}"
                )
            ),
        )
    )

    required_runtime_assets = _selected_runtime_assets(asset_root, formal_episodes)
    missing_runtime_assets = [
        str(path) for path in required_runtime_assets if not path.is_file()
    ]
    checks.append(
        Verification(
            "Robot and Human runtime assets are present",
            not missing_runtime_assets,
            "present"
            if not missing_runtime_assets
            else ", ".join(missing_runtime_assets),
        )
    )

    checkpoint_variables = (
        "TARSPHERE_SAM3_CHECKPOINT",
        "TARSPHERE_XNAVDP_CHECKPOINT",
        "TARSPHERE_FRONTIERNET_CHECKPOINT",
    )
    invalid_checkpoints = []
    for variable in checkpoint_variables:
        value = os.environ.get(variable, "").strip()
        if not value or not Path(value).expanduser().is_file():
            invalid_checkpoints.append(f"{variable}={value or 'unset'}")
    checks.append(
        Verification(
            "SAM3, X-NavDP, and FrontierNet checkpoints are present",
            not invalid_checkpoints,
            "present" if not invalid_checkpoints else "; ".join(invalid_checkpoints),
        )
    )

    grasp_root = (
        REPOSITORY_ROOT
        / "third_party"
        / "GraspGenX"
        / "ext"
        / "graspgenx_checkpoints"
        / "release"
    )
    grasp_assets = (
        REPOSITORY_ROOT
        / "third_party"
        / "GraspGenX"
        / "ext"
        / "gripper_descriptions"
        / "gripper_descriptions"
        / "assets"
    )
    checks.append(
        Verification(
            "GraspGenX checkpoint and gripper assets are present",
            all(
                (
                    (grasp_root / "gen" / "config.yaml").is_file(),
                    (grasp_root / "dis" / "config.yaml").is_file(),
                    bool(list((grasp_root / "gen").glob("epoch_*.pth"))),
                    bool(list((grasp_root / "dis").glob("epoch_*.pth"))),
                    grasp_assets.is_dir(),
                )
            ),
            f"checkpoints={grasp_root}; grippers={grasp_assets}",
        )
    )

    if asset_root.is_dir():
        free_storage_gib = shutil.disk_usage(asset_root).free / (1024**3)
        checks.append(
            Verification(
                "Asset filesystem has room for caches and initial outputs",
                free_storage_gib >= MINIMUM_FREE_STORAGE_GIB,
                f"{free_storage_gib:.1f} GiB free",
                required=False,
            )
        )

    vulkan_icd = os.environ.get("VK_ICD_FILENAMES", "").strip()
    vulkan_icd_paths = [Path(item) for item in vulkan_icd.split(os.pathsep) if item]
    checks.append(
        Verification(
            "NVIDIA Vulkan ICD is available",
            bool(vulkan_icd_paths) and all(path.is_file() for path in vulkan_icd_paths),
            vulkan_icd or "VK_ICD_FILENAMES is unset",
        )
    )

    checks.append(
        Verification(
            "ffmpeg is available for optional evaluation videos",
            shutil.which("ffmpeg") is not None,
            shutil.which("ffmpeg") or "not found",
            required=False,
        )
    )

    print("TARSphere installation preflight")
    for check in checks:
        marker = "PASS" if check.passed else "FAIL" if check.required else "WARN"
        print(f"[{marker}] {check.description}: {check.observed}")
    failures = [check for check in checks if check.required and not check.passed]
    if failures:
        print(f"\nPreflight failed: {len(failures)} condition(s) must be fixed.")
        return 1
    print(
        "\nPreflight passed. Run scripts/evaluate.sh; runtime capacity requires a real episode smoke test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
