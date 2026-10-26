"""Run the real G1 runtime in an empty scene, without model APIs or GRScenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--gpu", type=int, default=0, help="Physical NVIDIA GPU index")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator

    configure_cuda_allocator()

    from tarsphere.infra.simulation.demo_gpu import lookup_physical_gpu_uuid

    gpu_uuid = lookup_physical_gpu_uuid(args.gpu)
    os.environ["TARSPHERE_KIT_ACTIVE_GPU"] = str(args.gpu)
    os.environ["TARSPHERE_DEMO_GPU_UUID"] = gpu_uuid
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

    import numpy as np
    from PIL import Image

    from tarsphere.bench.schema import TARSphereEpisode
    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv
    from tarsphere.infra.simulation.isaac_process_lock import acquire_isaac_process_lock

    scene = Path(__file__).resolve().parents[2] / "assets/scenes/g1_camera_smoke.usda"
    episode = TARSphereEpisode.model_validate(
        {
            "episode_id": "g1_camera_smoke",
            "scene": {"asset_path": str(scene)},
            "robot": {
                "id": 0,
                "model": "G1Robot",
                "start_position": [0, 0, 0.8],
                "start_orientation": [1, 0, 0, 0],
            },
        }
    )
    # Preflight belongs in a separate process: loading UsdSchemaRegistry before
    # SimulationApp registers PhysX schemas can abort the native runtime.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from types import SimpleNamespace; "
            "from tarsphere.assets import tarsphere_assets_root; "
            "from tarsphere.cli.check_environment import _robot_runtime_asset_issues; "
            "issues = _robot_runtime_asset_issues(tarsphere_assets_root(), "
            "[SimpleNamespace(robot=SimpleNamespace(robot_type='G1Robot'))]); "
            "assert not issues, issues",
        ],
        check=True,
        timeout=120,
    )
    runtime_lock = acquire_isaac_process_lock("G1 camera smoke")
    env = IsaacAutonomousEnv(
        episode,
        headless=True,
        enable_streaming=False,
        enable_human_ego_cameras=False,
        image_dir=str(output),
        viewer_mode="first_person",
    )
    try:
        frames = []
        # Exercise the same cold preparation and resident soft reset as workers.
        observations = [env.reset(scene_prepare_only=True, skip_navmesh=True)]
        env.load_episode(episode)
        observations.append(env.reset(skip_navmesh=True))
        for _ in range(args.steps):
            env._obs, _, _, _, _ = env._env.step(
                action={"move_by_speed": [0.0, 0.0, 0.0]}
            )
            env._physics_steps += 1
        observations.append(env.observe())
        env._assert_robot_upright(
            action_name="camera_smoke", env_action={"move_by_speed": [0.0, 0.0, 0.0]}
        )
        from tarsphere.infra.simulation.isaac_env import camera_local_orientation_wxyz

        camera_prim = env._require_robot_camera_prim(env._agent_camera_path())
        measured_orientation = env._camera_local_orientation_wxyz(camera_prim)
        expected_orientation = camera_local_orientation_wxyz(
            env._camera_rest_local_quat, -15.0
        )
        assert np.allclose(measured_orientation, expected_orientation, atol=1e-6)
        for index, observation in enumerate(observations):
            rgb = np.asarray(observation["rgb"])
            depth = np.asarray(observation["depth"])
            assert rgb.shape[:2] == (480, 640), rgb.shape
            assert depth.shape[:2] == (480, 640), depth.shape
            assert float(rgb[..., :3].std()) > 1.0, "RGB is blank"
            assert observation["depth_status"] == "valid", observation["depth_status"]
            assert observation["camera_path"].endswith("/d435_link/Camera")
            assert abs(float(observation["camera_params"]["fov_deg"]) - 100) < 1e-6
            assert abs(float(observation["pose"]["camera_pitch_deg"]) + 15) < 1e-6
            Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
                output / f"rgb_{index}.png"
            )
            np.save(output / f"depth_{index}.npy", depth)
            frames.append(
                {
                    "rgb_shape": list(rgb.shape),
                    "depth_shape": list(depth.shape),
                    "depth_status": observation["depth_status"],
                    "depth_valid_ratio": observation.get("depth_valid_ratio"),
                    "camera_path": observation["camera_path"],
                    "fov_deg": observation["camera_params"]["fov_deg"],
                    "mount_pitch_deg": observation["pose"]["camera_pitch_deg"],
                    "position": observation["pose"]["position"],
                }
            )
        report = {
            "passed": True,
            "physical_gpu": args.gpu,
            "gpu_uuid": gpu_uuid,
            "physics_steps": args.steps,
            "frames": frames,
            "camera_rest_orientation_wxyz": list(env._camera_rest_local_quat),
            "camera_live_orientation_wxyz": list(measured_orientation),
            "camera_expected_orientation_wxyz": list(expected_orientation),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    except BaseException as error:
        (output / "report.json").write_text(
            json.dumps(
                {
                    "passed": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit's skip-cleanup shutdown exits with zero even while unwinding an
        # exception. Preserve the failed diagnostic's real process status.
        os._exit(1)
    env.close(skip_kit_cleanup=True)
    runtime_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
