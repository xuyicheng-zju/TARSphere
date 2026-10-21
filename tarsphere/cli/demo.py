#!/usr/bin/env python3
"""Plan, record, and verify native-resolution public demo highlights.

The model receives 640x480 RGB; depth remains private to tools. An independent camera follows its pose and optics
at 2560x1920; third-person cameras record 2560x1440.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tarsphere.evaluation.official import FORMAL_EVALUATION_MAX_DECISION_STEPS
from tarsphere.provider_compat import add_json_request_controls

# Website identifiers map to the benchmark's authoritative directory vocabulary.
CONDITIONS = {
    "base": "base",
    "pr": "perceptual_robustness",
    "da": "disruption_adaptation",
    "tu": "task_update",
    "hc": "human_coordination",
    "iu": "intent_understanding",
}
FAMILIES = {
    "navigate_and_report": "navigate_and_report",
    "fetch_and_deliver": "fetch_and_deliver",
}
# Keep scene-local case ordering deterministic for recording and provenance.
CASE_EXECUTION_ORDER = [
    f"{family}-{condition}"
    for family in FAMILIES
    for condition in ("base", "tu", "pr", "da", "hc", "iu")
]
# Presentation streams never enter model perception or grounding.
CAPTURE = {
    "ego_resolution": [640, 480],
    "presentation_ego_resolution": [2560, 1920],
    "observer_resolution": [2560, 1440],
    "composition_resolution": [2560, 1440],
    "fps": 30,
    "observer_views": ["presentation_ego", "follow"],
    "physics_stride": 8,
    "camera_contract": "independent_presentation_camera",
    "model_observation_resolution": [640, 480],
    "human_motion_backend": "animation",
    "navmesh_bake_backend": "native_cpu",
    # Apply the same bounded policy budget to every demo condition; the frozen
    # episode files intentionally leave this override to the launcher.
    "max_model_decisions": FORMAL_EVALUATION_MAX_DECISION_STEPS,
    "clock": "simulation_time; inference wall time is not video time",
}
DEBUG_CAPTURE = {
    **CAPTURE,
    "presentation_ego_resolution": [640, 480],
    "observer_resolution": [640, 360],
    "purpose": "debug_only; no publication composition or verification",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def default_cases() -> list[dict]:
    """Return the source examples available to the recording launcher."""
    return [
        {
            "id": f"{family}-{condition}",
            "episode": f"episodes/case/{family}/{name}.yaml",
        }
        for family in FAMILIES
        for condition, name in CONDITIONS.items()
    ]


def load_cases(config: Path | None = None) -> list[dict]:
    """Bind recording slots to user episodes or the source examples."""
    data = (
        json.loads(config.read_text())
        if config is not None
        else {"cases": default_cases()}
    )
    if set(data) != {"cases"}:
        raise ValueError("Demo configuration must contain only cases.")
    expected = {
        f"{family}-{condition}" for family in FAMILIES for condition in CONDITIONS
    }
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("Demo configuration requires a nonempty cases list.")
    cases = []
    seen = set()
    scene_names = {}
    for item in data["cases"]:
        if set(item) != {"id", "episode"}:
            raise ValueError("Each demo case requires exactly id and episode.")
        key = item["id"]
        if key not in expected or key in seen:
            raise ValueError(f"Unknown or duplicate demo slot: {key}")
        seen.add(key)
        path = ROOT / item["episode"]
        path.resolve(strict=True)
        episode = yaml.safe_load(path.read_text())
        if any(
            human.get("scene_person_prim_path") or not human.get("asset")
            for human in episode["humans"]
        ):
            raise ValueError(
                f"{key}: animated public demos require rigged catalog actors."
            )
        family, condition = key.split("-")
        profile = episode["evaluation"]["task_profile"]
        if (profile["mainline"], profile["capability_slice"]) != (
            FAMILIES[family],
            CONDITIONS[condition],
        ):
            raise ValueError(f"{key} is bound to the wrong task family or condition.")
        scene_asset = episode["scene"]["asset_path"]
        scene_name = Path(scene_asset).parent.name
        if scene_name in scene_names and scene_names[scene_name] != scene_asset:
            raise ValueError(
                f"{scene_name}: different scene assets would share an output directory."
            )
        scene_names[scene_name] = scene_asset
        cases.append(
            {
                **item,
                "episode_id": episode["episode_id"],
                "episode_sha256": digest(path),
                "scene_asset": scene_asset,
                "scene_name": scene_name,
            }
        )
    return cases


def case_groups(cases: list[dict]) -> list[list[dict]]:
    groups = {}
    for case in sorted(cases, key=lambda item: CASE_EXECUTION_ORDER.index(item["id"])):
        groups.setdefault(case["scene_asset"], []).append(case)
    return list(groups.values())


def case_takes(session: Path, case: dict) -> list[Path]:
    return [
        take
        for take in sorted(
            (session / "scenes" / case["scene_name"]).glob("take-[0-9][0-9][0-9]")
        )
        if case["id"] in json.loads((take / "launch.json").read_text())["cases"]
    ]


def worker_command(
    cases: list[dict], output: Path, model: str, *, capture: dict = CAPTURE
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "tarsphere/cli/worker.py"),
        "--model",
        model,
        "--output-dir",
        str(output),
        "--headless",
        "--render-mode",
        "demo",
        "--record-video",
        "--evaluate-after-rollout",
        "--max-steps",
        str(capture["max_model_decisions"]),
        "--human-motion-backend",
        "animation",
        "--room-observer-width",
        str(capture["observer_resolution"][0]),
        "--room-observer-height",
        str(capture["observer_resolution"][1]),
        "--room-observer-stride",
        "8",
    ]
    for case in cases:
        command += ["--episode", str(ROOT / case["episode"])]
    for view in capture["observer_views"]:
        command += ["--room-observer-view", view]
    return command


def capture_environment(
    gpu: int,
    navdp_port: int,
    *,
    isolated_gpu: bool = False,
    runtime_slot: int = 0,
) -> dict[str, str]:
    if gpu < 0 or runtime_slot < 0:
        raise ValueError("Physical GPU and runtime slot must be nonnegative.")
    env = dict(os.environ)
    # Isaac indexes physical GPUs; its isolated model workers use a separate mask.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("TARSPHERE_DEMO_GPU_UUID", None)
    env.pop("TARSPHERE_SCHEDULER_PID", None)
    env.pop("TARSPHERE_KEEP_WORKER_ALIVE", None)
    env.update(
        {
            "TARSPHERE_KIT_ACTIVE_GPU": str(gpu),
            "TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES": str(gpu),
            "TARSPHERE_ISAAC_RUNTIME_SLOT": str(runtime_slot),
            "TARSPHERE_MODEL_GPU": str(gpu),
            "TARSPHERE_FRONTIERNET_DEVICE": f"cuda:{gpu}",
            "TARSPHERE_XNAVDP_PORT": str(navdp_port),
            "PYTHONUNBUFFERED": "1",
            "TARSPHERE_SCHEDULER_PID": str(os.getpid()),
            "TARSPHERE_ROBOT_CAMERA_WIDTH": "640",
            "TARSPHERE_ROBOT_CAMERA_HEIGHT": "480",
            "TARSPHERE_PRESENTATION_CAPTURE": "1",
            # This native MeshTools build allocates on CUDA device 0 independently
            # of Kit's physical GPU selection. Bake once on CPU, reuse all cases.
            "TARSPHERE_NAVMESH_USE_GPU": "0",
        }
    )
    if isolated_gpu:
        from tarsphere.infra.simulation.demo_gpu import (
            lookup_physical_gpu_uuid,
            resolve_demo_gpu_mapping,
        )

        gpu_uuid = lookup_physical_gpu_uuid(gpu)
        env.update(
            {
                "TARSPHERE_DEMO_GPU_UUID": gpu_uuid,
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES": gpu_uuid,
                "TARSPHERE_MODEL_GPU": "0",
                "TARSPHERE_FRONTIERNET_DEVICE": "cuda:0",
                "TARSPHERE_NAVMESH_USE_GPU": "1",
            }
        )
        # Validate physical identity without importing Torch or creating a
        # CUDA context before Isaac. Kit bootstrap repeats this check.
        resolve_demo_gpu_mapping(env)
    return env


def check_model_transport(output: Path, model: str) -> dict:
    """Require a real SDK response before paying the scene initialization cost."""
    from openai import OpenAI

    receipt = {
        "requested_model": model,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    started = time.monotonic()
    try:
        request = {
            "model": model,
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": 'Return only the JSON object {"ok": true}.',
                },
            ],
        }
        add_json_request_controls(request, model=model, temperature=0.0)
        with OpenAI(
            api_key=os.environ["TARSPHERE_AGENT_API_KEY"],
            base_url=os.environ["TARSPHERE_AGENT_API_BASE"],
            timeout=30.0,
            max_retries=0,
        ) as client:
            response = client.chat.completions.create(**request)
        if json.loads(response.choices[0].message.content) != {"ok": True}:
            raise ValueError("The model transport probe returned invalid JSON.")
        provider_model = response.model
        receipt["provider_model"] = provider_model
        if provider_model != model and not provider_model.startswith(model + "-"):
            raise ValueError("The endpoint returned a different model identity.")
        receipt.update(
            status="passed",
            provider_model=provider_model,
            usage=response.usage.model_dump() if response.usage else None,
        )
    except Exception as error:
        # Provider exception strings can contain request headers. Keep only
        # the exception class, retain the failure receipt, and stop the launch.
        receipt.update(status="failed", error_type=type(error).__name__)
        raise OSError(
            f"Model transport preflight failed ({type(error).__name__}); "
            f"inspect {output / 'transport-preflight.json'}."
        ) from None
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "transport-preflight.json", receipt)
    return receipt


def checked_file(value: str, root: Path) -> Path:
    path = Path(value).resolve(strict=True)
    path.relative_to(root.resolve())
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty capture artifact: {path}")
    return path


def probe_video(path: Path, resolution: list[int]) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if [stream["width"], stream["height"]] != resolution:
        raise ValueError(
            f"{path.name}: native/output resolution does not match {resolution}."
        )
    if denominator == 0 or abs(numerator / denominator - CAPTURE["fps"]) > 0.001:
        raise ValueError(f"{path.name}: expected a 30 fps simulation-time recording.")
    if float(info["format"]["duration"]) <= 0:
        raise ValueError(f"{path.name}: empty video.")
    return {**stream, "duration": info["format"]["duration"], "sha256": digest(path)}


def verify_frame_clock(rows: list[dict], frame_count: int, view: str) -> None:
    sources = [row["physics_step"] for row in rows]
    ticks = [row["video_physics_step"] for row in rows]
    if (
        not sources
        or any(b < a for a, b in zip(sources, sources[1:]))
        or any(b - a != CAPTURE["physics_stride"] for a, b in zip(ticks, ticks[1:]))
        or any(source > tick for source, tick in zip(sources, ticks))
        or frame_count != len(rows)
    ):
        raise ValueError(
            f"{view}: frames must follow the causal 30 fps physics-time grid."
        )


def verify_capture(
    output: Path, case: dict, model: str, *, capture: dict = CAPTURE
) -> dict:
    result_path = checked_file(str(output / f"{case['episode_id']}.json"), output)
    result = json.loads(result_path.read_text())
    if result["episode_id"] != case["episode_id"]:
        raise ValueError("Captured episode identity does not match the selected slot.")
    if result.get("outcome_status") == "infrastructure_invalid" or result.get(
        "observer_video_error"
    ):
        raise ValueError(
            "Capture has an infrastructure or observer-video error; inspect worker.log."
        )
    models = list(result.get("agent_model_usage", {}).get("by_model", {}))
    if models != [model]:
        raise ValueError(
            f"Recorded model identity {models!r} does not match {model!r}."
        )
    videos = result["continuous_ego_videos"]
    required = {
        "ego": (videos["ego"], capture["ego_resolution"]),
    }
    for view in capture["observer_views"]:
        resolution = (
            capture["presentation_ego_resolution"]
            if view == "presentation_ego"
            else capture["observer_resolution"]
        )
        required[view] = (result["observer_videos"][view]["path"], resolution)
    media = {
        name: probe_video(checked_file(path, output), resolution)
        for name, (path, resolution) in required.items()
    }
    for view in capture["observer_views"]:
        manifest = checked_file(
            result["observer_videos"][view]["frame_manifest"], output
        )
        rows = [
            json.loads(line)
            for line in manifest.read_text().splitlines()
            if line.strip()
        ]
        verify_frame_clock(rows, int(media[view]["nb_frames"]), view)
    for key in [
        "heartbeat_path",
        "model_output_trace_path",
        "rollout_artifact_path",
        "episode_artifact_path",
    ]:
        checked_file(result[key], output)
    frames = checked_file(videos["frame_manifest"], output)
    frame_rows = [
        json.loads(line) for line in frames.read_text().splitlines() if line.strip()
    ]
    verify_frame_clock(frame_rows, int(media["ego"]["nb_frames"]), "ego")
    slug = frames.name.removesuffix("_ego_frames.jsonl")
    inputs = output / f"{slug}.model_output.inputs"
    if not any(inputs.glob("*.request.json")):
        raise ValueError(f"Matching model-input snapshots are missing: {inputs}")
    presentation = output / "online_maps" / slug / "presentation"
    for folder, pattern in [("rgbd", "*.npz"), ("memory", "*.json")]:
        if not any((presentation / folder).glob(pattern)):
            raise ValueError(
                f"Required {folder} presentation evidence was not captured."
            )
    panel = presentation_panel_for_case(output, case, result)
    if panel != "observed_spatial_memory":
        load_presentation_sensor_evidence(output, result, panel)
    # A captured failure is a valid demo source. Capture completeness never means success.
    return {
        "case": case["id"],
        "capture_complete": True,
        "task_success": result.get("success"),
        "outcome_status": result.get("outcome_status"),
        "termination_reason": result.get("termination_reason"),
        "media": media,
        "camera_contract": capture["camera_contract"],
        "capture_purpose": capture.get("purpose", "public_demo"),
        "result_sha256": digest(result_path),
    }


def presentation_panel_for_case(output: Path, case: dict, result: dict) -> str:
    from tarsphere.infra.perception.demo_compositor import recorded_sensor_panel

    panel = recorded_sensor_panel(result=result, result_dir=output)
    if (panel != "observed_spatial_memory") != case["id"].endswith("-pr"):
        raise ValueError(
            "Presentation case condition differs from its immutable episode artifact."
        )
    return panel


def load_presentation_sensor_evidence(output: Path, result: dict, panel: str):
    from tarsphere.infra.perception.demo_compositor import (
        load_conditioned_depth_observations,
        load_model_observations,
    )

    if panel == "grounding_depth":
        return load_conditioned_depth_observations(result=result, result_dir=output)
    if panel == "model_observation":
        slug = Path(
            result["continuous_ego_videos"]["frame_manifest"]
        ).name.removesuffix("_ego_frames.jsonl")
        return load_model_observations(
            result=result,
            result_dir=output,
            model_input_dir=output / f"{slug}.model_output.inputs",
        )
    raise ValueError("This presentation panel is not a recorded sensor view.")


def render_capture(output: Path, case: dict) -> Path:
    """Compose one verified recording; never change or rerun its physical trace."""
    result_path = output / f"{case['episode_id']}.json"
    result = json.loads(result_path.read_text())
    video = output / "presentation" / case["id"] / "demo-2k.mp4"
    if video.exists():
        verify_presentation(output, case)
        return video
    observer = result["observer_videos"]
    slug = Path(result["continuous_ego_videos"]["frame_manifest"]).name.removesuffix(
        "_ego_frames.jsonl"
    )
    command = [
        sys.executable,
        str(ROOT / "tarsphere/cli/render.py"),
        "--result",
        str(result_path),
        "--ego",
        observer["presentation_ego"]["path"],
        "--ego-frames",
        observer["presentation_ego"]["frame_manifest"],
        "--observer",
        observer["follow"]["path"],
        "--observer-frames",
        observer["follow"]["frame_manifest"],
        "--heartbeats",
        result["heartbeat_path"],
        "--online-map-dir",
        str(output / "online_maps" / slug),
        "--output",
        str(video),
        "--height",
        "1440",
    ]
    panel = presentation_panel_for_case(output, case, result)
    if panel == "model_observation":
        command += [
            "--show-model-observation",
            "--model-input-dir",
            str(output / f"{slug}.model_output.inputs"),
        ]
    elif panel == "grounding_depth":
        command += ["--show-grounding-depth"]
    subprocess.run(command, cwd=ROOT, check=True)
    verify_presentation(output, case)
    return video


def verify_presentation(output: Path, case: dict) -> dict:
    video = checked_file(
        str(output / "presentation" / case["id"] / "demo-2k.mp4"), output
    )
    metadata = json.loads(video.with_suffix(".presentation.json").read_text())
    if not metadata["synchronization_verified"]:
        raise ValueError(
            "An archive layout preview cannot pass as a synchronized demo."
        )
    if metadata["source_result_sha256"] != digest(
        output / f"{case['episode_id']}.json"
    ):
        raise ValueError("Presentation belongs to a different rollout result.")
    if metadata["video_sha256"] != digest(video):
        raise ValueError("Presentation video changed after composition.")
    manifest = checked_file(metadata["frame_manifest"], output)
    if metadata["frame_manifest_sha256"] != digest(manifest):
        raise ValueError("Presentation timeline changed after composition.")
    result = json.loads((output / f"{case['episode_id']}.json").read_text())
    rollout = metadata["source_rollout"]
    if (
        rollout["path"] != result["rollout_artifact_path"]
        or rollout["sha256"] != result["rollout_artifact_sha256"]
        or digest(checked_file(rollout["path"], output)) != rollout["sha256"]
    ):
        raise ValueError("Presentation trace no longer matches the recorded rollout.")
    if metadata["source_heartbeats_sha256"] != digest(
        checked_file(result["heartbeat_path"], output)
    ):
        raise ValueError("Presentation heartbeats no longer match the recording.")
    observers = result["observer_videos"]
    expected_sources = {
        "input_videos": [
            observers[view]["path"] for view in ("presentation_ego", "follow")
        ],
        "input_frame_manifests": [
            observers[view]["frame_manifest"] for view in ("presentation_ego", "follow")
        ],
    }
    for paths_key, hashes_key in [
        ("input_videos", "input_video_sha256"),
        ("input_frame_manifests", "input_frame_manifest_sha256"),
    ]:
        if metadata[paths_key] != expected_sources[paths_key] or len(
            metadata[hashes_key]
        ) != len(expected_sources[paths_key]):
            raise ValueError(
                "Presentation does not use the selected take and camera views."
            )
        for path, expected in zip(metadata[paths_key], metadata[hashes_key]):
            if digest(checked_file(path, output)) != expected:
                raise ValueError("Presentation source media changed after composition.")
    panel = presentation_panel_for_case(output, case, result)
    if metadata["right_panel"] != panel:
        raise ValueError(
            "Presentation panel differs from the recorded sensor condition."
        )
    if panel in {"model_observation", "grounding_depth"}:
        captures, evidence_rollout = load_presentation_sensor_evidence(
            output, result, panel
        )
        if metadata[f"{panel}_rollout"] != evidence_rollout or metadata[
            f"{panel}_evidence"
        ] != [capture.provenance() for capture in captures]:
            raise ValueError(
                "PR panel evidence no longer matches the exact recorded sensor captures."
            )
    return probe_video(video, CAPTURE["composition_resolution"])


def clean_ego_presentation(output: Path, case: dict, capture_report: dict) -> dict:
    """Publish the verified native ego source directly, with its original clock."""
    result_path = output / f"{case['episode_id']}.json"
    if capture_report["result_sha256"] != digest(result_path):
        raise ValueError("Clean ego result changed after capture verification.")
    result = json.loads(result_path.read_text())
    source = result["observer_videos"]["presentation_ego"]
    video = checked_file(source["path"], output)
    if capture_report["media"]["presentation_ego"]["sha256"] != digest(video):
        raise ValueError("Clean ego video changed after capture verification.")
    return {
        "layout": "clean_ego",
        "path": str(video),
        "frame_manifest": str(checked_file(source["frame_manifest"], output)),
        "media": capture_report["media"]["presentation_ego"],
    }


def main() -> int:
    if sys.argv[1:2] == ["teleop"]:
        from tarsphere.cli.teleop import main as run_teleop

        run_teleop(sys.argv[2:])
        return 0
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Interactive control: scripts/demo.sh teleop --help",
    )
    parser.add_argument("action", choices=["plan", "record", "render", "verify"])
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Nonempty demo manifest with unique slots; episodes may use different scenes.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Website slot, repeatable; default all configured cases.",
    )
    parser.add_argument(
        "--model",
        default="gpt-6.1-sol",
        help="Exact gateway model name for this new demo campaign.",
    )
    parser.add_argument(
        "--session", default="demo", help="Name for this recording session."
    )
    parser.add_argument(
        "--gpu", type=int, help="Physical GPU index; required for recording."
    )
    parser.add_argument(
        "--navdp-port",
        type=int,
        help="Dedicated, already running local X-NavDP service; no concurrent worker may use it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip verified captures; keep failed attempts before retrying.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Plan/record only: clean cameras at 640 width, without publication rendering.",
    )
    parser.add_argument(
        "--ego-only",
        action="store_true",
        help="Record only the independent clean ego camera; omit third-person cameras and composition.",
    )
    parser.add_argument(
        "--isolated-gpu",
        action="store_true",
        help="Explicit demo mode: one GPU UUID; renderer uses its physical index, CUDA uses ordinal 0.",
    )
    parser.add_argument(
        "--runtime-slot",
        type=int,
        default=0,
        help="Independent Isaac runtime/cache slot (nonnegative; default 0).",
    )
    args = parser.parse_args()
    if args.debug and args.action not in {"plan", "record"}:
        parser.error(
            "--debug is for plan/record; publication render/verify require native 2K sources."
        )
    capture = dict(DEBUG_CAPTURE if args.debug else CAPTURE)
    if args.ego_only:
        capture.update(
            observer_views=["presentation_ego"], presentation_layout="clean_ego"
        )
    if args.isolated_gpu:
        capture.update(
            navmesh_bake_backend="native_gpu",
            gpu_mapping="single_uuid; renderer_physical_index; cuda_ordinal_zero",
        )
    if args.runtime_slot < 0:
        parser.error("--runtime-slot must be nonnegative.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.session):
        parser.error("--session must be a simple name without path separators.")
    if args.action == "record" and (args.gpu is None or args.gpu < 0):
        parser.error("Recording requires an explicitly assigned non-negative --gpu.")
    if args.action == "record" and (
        args.navdp_port is None or not 1 <= args.navdp_port <= 65535
    ):
        parser.error(
            "Recording requires an explicitly assigned --navdp-port in 1..65535."
        )
    all_cases = load_cases(args.config)
    known = {case["id"] for case in all_cases}
    if len(args.case) != len(set(args.case)) or set(args.case) - known:
        parser.error(f"Choose unique cases from: {', '.join(sorted(known))}")
    cases = [case for case in all_cases if not args.case or case["id"] in args.case]
    session = Path.home() / ".local/share/tarsphere/demos" / args.session
    contract = {"model": args.model, "capture": capture, "cases": all_cases}
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "contract": contract,
                    "selected": [c["id"] for c in cases],
                    "output": str(session),
                    "commands": [
                        worker_command(
                            group,
                            session / "scenes" / group[0]["scene_name"] / "take-001",
                            args.model,
                            capture=capture,
                        )
                        for group in case_groups(cases)
                    ],
                },
                indent=2,
            )
        )
        return 0
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        parser.error("FFmpeg and ffprobe are required.")
    if args.action == "record" and (
        sys.platform != "linux" or sys.version_info < (3, 10)
    ):
        parser.error(
            "Record on the configured Linux Isaac host with Python >= 3.10; plan/verify work locally."
        )
    if args.action == "record":
        for name in ["TARSPHERE_AGENT_API_BASE", "TARSPHERE_AGENT_API_KEY"]:
            if not os.environ.get(name, "").strip():
                parser.error(f"Set {name} explicitly for the demo model endpoint.")
        # The service is intentionally not borrowed from an active evaluation lane.
        with socket.create_connection(("127.0.0.1", args.navdp_port), timeout=3):
            pass
    contract_path = session / "capture-contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "Session model, capture settings, or episode hashes changed. Choose a new session."
            )
    elif args.action in {"verify", "render"}:
        raise FileNotFoundError(f"No capture contract: {contract_path}")
    else:
        session.mkdir(parents=True, exist_ok=True)
        write_json(contract_path, contract)
    if args.action in {"verify", "render"}:
        for case in cases:
            takes = case_takes(session, case)
            if not takes:
                raise FileNotFoundError(f"No capture for {case['id']}")
            report = verify_capture(takes[-1], case, args.model, capture=capture)
            if args.ego_only:
                report["presentation"] = clean_ego_presentation(takes[-1], case, report)
            elif args.action == "render":
                render_capture(takes[-1], case)
                report["presentation"] = verify_presentation(takes[-1], case)
            else:
                report["presentation"] = verify_presentation(takes[-1], case)
            print(json.dumps(report, ensure_ascii=False))
        return 0
    pending = []
    for case in cases:
        takes = case_takes(session, case)
        if not takes:
            pending.append(case)
            continue
        if not args.resume:
            raise FileExistsError(
                f"{case['id']} already has captures. Use --resume or a new --session."
            )
        try:
            verify_capture(takes[-1], case, args.model, capture=capture)
        except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError):
            print(
                f"{case['id']}: preserving the incomplete take and retrying.",
                flush=True,
            )
            pending.append(case)
        else:
            if not args.debug and not args.ego_only:
                render_capture(takes[-1], case)
            print(f"{case['id']}: capture verified, skipping.", flush=True)
    for group in case_groups(pending):
        scene_name = group[0]["scene_name"]
        directory = session / "scenes" / scene_name
        previous = sorted(directory.glob("take-[0-9][0-9][0-9]"))
        number = int(previous[-1].name[-3:]) + 1 if previous else 1
        output = directory / f"take-{number:03d}"
        output.mkdir(parents=True, exist_ok=False)
        command = worker_command(group, output, args.model, capture=capture)
        worker_environment = capture_environment(
            args.gpu,
            args.navdp_port,
            isolated_gpu=args.isolated_gpu,
            runtime_slot=args.runtime_slot,
        )
        write_json(
            output / "launch.json",
            {
                "command": command,
                "physical_gpu": args.gpu,
                "runtime_slot": args.runtime_slot,
                "gpu_uuid": worker_environment.get("TARSPHERE_DEMO_GPU_UUID"),
                "navdp_port": args.navdp_port,
                "capture": capture,
                "cases": [case["id"] for case in group],
            },
        )
        check_model_transport(output, args.model)
        print(
            f"{scene_name}: {len(group)} cases share one scene load on GPU {args.gpu}; log: {output / 'worker.log'}",
            flush=True,
        )
        with (output / "worker.log").open("wb") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=worker_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        reports = []
        for case in group:
            report = verify_capture(output, case, args.model, capture=capture)
            if args.ego_only:
                report["presentation"] = clean_ego_presentation(output, case, report)
            elif not args.debug:
                render_capture(output, case)
                report["presentation"] = verify_presentation(output, case)
            reports.append(report)
            write_json(output / "capture-report.json", reports)
            print(
                f"{case['id']}: capture verified; task success={report['task_success']}.",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Demo error: {error}", file=sys.stderr)
        raise SystemExit(1)
