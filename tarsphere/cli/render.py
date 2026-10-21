#!/usr/bin/env python3
"""Render clean ego/third-person recordings with shared evidence panels below."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tarsphere.infra.perception.demo_compositor import (
    load_conditioned_depth_observations,
    load_model_observations,
    load_recorded_rollout,
    recorded_sensor_panel,
    render_demo_presentation,
    _recorded_artifact_path,
    file_sha256,
    load_editorial_annotations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--source-result-root",
        type=Path,
        help="Original absolute remote take directory; recorded artifacts and PR sensor evidence map read-only into the local --result directory",
    )
    parser.add_argument("--ego", type=Path, required=True)
    parser.add_argument("--ego-frames", type=Path, required=True)
    parser.add_argument("--observer", type=Path)
    parser.add_argument("--observer-frames", type=Path)
    parser.add_argument("--heartbeats", type=Path, required=True)
    parser.add_argument("--online-map-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-title", default="")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Independent hash-bound author review sidecar; exact recorded thinking and action clock only",
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--display-grade",
        choices=["none", "neutral", "indoor"],
        default="none",
        help="Display-only ego colour curve; original camera and model input bytes remain unchanged",
    )
    parser.add_argument(
        "--preserve-map",
        action="store_true",
        help="Full HUD: display the complete original memory map without extra BEV badges; PR model-input verification remains required",
    )
    parser.add_argument(
        "--ego-frontier-labels",
        action="store_true",
        help="Native ego frontier letters seeded only at exact captured cameras; uncertain image-space matches are hidden",
    )
    parser.add_argument(
        "--ego-frontier-cache",
        type=Path,
        help="Hash-bound receipt for reviewed native ASS frontier points; reuse source tracking without recomputing NCC",
    )
    parser.add_argument(
        "--display-affine-manifest",
        type=Path,
        help="Native display stabilization sidecar; original camera/clock remain evidence, UI is composed after the warp",
    )
    sensor_panel = parser.add_mutually_exclusive_group()
    sensor_panel.add_argument(
        "--show-model-observation",
        action="store_true",
        help="PR: show the exact model request image in the shared right panel",
    )
    sensor_panel.add_argument(
        "--show-grounding-depth",
        action="store_true",
        help="depth_noise PR: show the actual conditioned depth visualization for grounding tools",
    )
    parser.add_argument(
        "--model-input-dir",
        type=Path,
        help="Matching take *.model_output.inputs directory; required for the RGB PR panel",
    )
    parser.add_argument(
        "--lidar-input-dir",
        type=Path,
        help="Full HUD radar: matching complete hash-bound *.model_output.inputs JSON archive; no image files needed",
    )
    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument(
        "--ego-full-hud",
        action="store_true",
        help="Native single ego with original floating cards and a right memory/action sidebar",
    )
    layout_group.add_argument(
        "--ego-only-layout",
        action="store_true",
        help="Keep native ego unscaled above the shared causal evidence panels; no observer input",
    )
    parser.add_argument(
        "--debug-640-preview",
        action="store_true",
        help="Explicit 640x480 single-ego preview, never a native 2K publication result",
    )
    parser.add_argument(
        "--policy-ego",
        action="store_true",
        help="PR: use the actual conditioned 640x480 robot sensor recording as ego; render HUD separately at full resolution",
    )
    parser.add_argument(
        "--terminal-hold-seconds",
        type=float,
        default=0.0,
        help="Single ego only: editorial repeat of the real terminal frame, with frozen source clock",
    )
    parser.add_argument(
        "--archive-layout-preview",
        action="store_true",
        help="Explicit unsynchronized low-resolution layout study; never a final demo",
    )
    args = parser.parse_args()
    single_ego = args.ego_only_layout or args.ego_full_hud
    if args.policy_ego and (
        not args.ego_full_hud
        or not args.preserve_map
        or args.debug_640_preview
        or args.archive_layout_preview
        or args.display_affine_manifest
        or args.ego_frontier_cache
        or args.display_grade != "none"
    ):
        parser.error(
            "--policy-ego requires full HUD and preserved map, without display grading, stabilization, native frontier cache, or preview modes"
        )
    if args.preserve_map and not args.ego_full_hud:
        parser.error("--preserve-map requires --ego-full-hud")
    if args.ego_frontier_labels and (not args.preserve_map or args.debug_640_preview):
        parser.error(
            "--ego-frontier-labels requires native --ego-full-hud --preserve-map"
        )
    if args.ego_frontier_cache and not args.ego_frontier_labels:
        parser.error("--ego-frontier-cache requires --ego-frontier-labels")
    if args.display_affine_manifest and (
        not args.ego_full_hud or args.debug_640_preview or args.archive_layout_preview
    ):
        parser.error(
            "--display-affine-manifest requires verified native --ego-full-hud"
        )
    if (
        args.display_affine_manifest
        and args.ego_frontier_labels
        and not args.ego_frontier_cache
    ):
        parser.error(
            "--display-affine-manifest with ego labels requires --ego-frontier-cache"
        )
    if args.annotations and (not args.ego_full_hud or args.archive_layout_preview):
        parser.error("--annotations requires the verified --ego-full-hud layout")
    if args.lidar_input_dir and not args.ego_full_hud:
        parser.error("--lidar-input-dir requires --ego-full-hud")
    if args.lidar_input_dir and args.archive_layout_preview:
        parser.error("Recorded LiDAR does not accept an unverified archive layout")
    if args.source_result_root and args.archive_layout_preview:
        parser.error("--source-result-root requires verified recorded evidence")
    if single_ego and (
        args.observer or args.observer_frames or args.archive_layout_preview
    ):
        parser.error("--ego-only-layout omits observer/archive inputs")
    if args.debug_640_preview and not single_ego:
        parser.error("--debug-640-preview requires --ego-only-layout")
    if not single_ego and not args.observer:
        parser.error("--observer is required for the default dual-camera layout")
    if not single_ego and not args.archive_layout_preview and not args.observer_frames:
        parser.error("--observer-frames is required for synchronized publication demos")
    if args.show_model_observation != bool(args.model_input_dir):
        parser.error(
            "--show-model-observation and --model-input-dir must be supplied together"
        )
    result_bytes = args.result.read_bytes()
    heartbeat_bytes = args.heartbeats.read_bytes()
    result = json.loads(result_bytes)
    source_rollout = None
    if args.archive_layout_preview:
        from tarsphere.infra.perception.trace_archive import normalize_archived_trace

        trace = normalize_archived_trace(result["trace"])
        human_replay_log = result.get("human_replay_log", [])
    else:
        rollout, source_rollout = load_recorded_rollout(
            result=result,
            result_dir=args.result.parent,
            **(
                {"source_result_root": args.source_result_root}
                if args.source_result_root
                else {}
            ),
        )
        requested_panel = (
            "grounding_depth"
            if args.show_grounding_depth
            else "model_observation"
            if args.show_model_observation
            else "observed_spatial_memory"
        )
        if requested_panel != recorded_sensor_panel(
            result=result,
            result_dir=args.result.parent,
            **(
                {"source_result_root": args.source_result_root}
                if args.source_result_root
                else {}
            ),
        ):
            raise ValueError(
                "Requested sensor panel does not match the recorded episode condition"
            )
        if (
            args.ego_full_hud
            and requested_panel == "model_observation"
            and not args.policy_ego
        ):
            parser.error(
                "Low-light and reduced-resolution publication requires --policy-ego and the original conditioned robot recording; an independent 2K camera would hide the perception condition"
            )
        trace = rollout["trace"]
        human_replay_log = rollout["human_replay_log"]
        if args.source_result_root and requested_panel != "observed_spatial_memory":
            # PR relocation changes only where original bytes are read. Do not
            # compose another take's camera or heartbeat under this evidence.
            def mapped(value):
                return _recorded_artifact_path(
                    value,
                    result_dir=args.result.parent,
                    source_result_root=args.source_result_root,
                )

            for view, video_arg, frames_arg in [
                ("presentation_ego", args.ego, args.ego_frames),
                *(
                    [("follow", args.observer, args.observer_frames)]
                    if args.observer
                    else []
                ),
            ]:
                recorded = (
                    dict(
                        path=result["continuous_ego_videos"]["ego"],
                        frame_manifest=result["continuous_ego_videos"][
                            "frame_manifest"
                        ],
                    )
                    if args.policy_ego and view == "presentation_ego"
                    else result["observer_videos"][view]
                )
                if video_arg.resolve() != mapped(
                    recorded["path"]
                ) or frames_arg.resolve() != mapped(recorded["frame_manifest"]):
                    raise ValueError(
                        "Relocated PR camera must match the recorded episode/attempt"
                    )
            if (
                args.heartbeats.resolve() != mapped(result["heartbeat_path"])
                or file_sha256(args.heartbeats) != result["heartbeat_sha256"]
            ):
                raise ValueError("Relocated PR heartbeats differ from recorded bytes")
            continuous = result["continuous_ego_videos"]
            for key, digest in result["continuous_ego_video_sha256"].items():
                if file_sha256(mapped(continuous[key])) != digest:
                    raise ValueError("Relocated PR continuous recording hash mismatch")
            slug = Path(continuous["frame_manifest"]).name.removesuffix(
                "_ego_frames.jsonl"
            )
            if (
                args.online_map_dir.resolve()
                != (args.result.parent / "online_maps" / slug).resolve()
            ):
                raise ValueError(
                    "Relocated PR mapping directory differs from the recorded attempt"
                )
            for record in trace:
                for call in record.get("model_calls", []):
                    path = mapped(call["input_snapshot"]["path"])
                    if (
                        args.lidar_input_dir
                        and path.parent != args.lidar_input_dir.resolve()
                    ):
                        raise ValueError(
                            "Relocated PR LiDAR input directory differs from its recorded request"
                        )
    beats = [json.loads(line) for line in heartbeat_bytes.splitlines() if line.strip()]
    policy_ego_evidence = None
    if args.policy_ego:
        from tarsphere.infra.perception.sensor_conditions import (
            REDUCED_RESOLUTION_SCALE,
            LOW_LIGHT_EXPOSURE_SCALE,
        )

        artifact = _recorded_artifact_path(
            result["episode_artifact_path"],
            result_dir=args.result.parent,
            source_result_root=args.source_result_root,
        )
        profile = json.loads(artifact.read_bytes())["evaluation"]["task_profile"]
        if profile["capability_slice"] != "perceptual_robustness":
            raise ValueError(
                "Policy ego publication requires a recorded perceptual condition"
            )
        continuous = result["continuous_ego_videos"]
        for key, supplied in [("ego", args.ego), ("frame_manifest", args.ego_frames)]:
            expected = _recorded_artifact_path(
                continuous[key],
                result_dir=args.result.parent,
                source_result_root=args.source_result_root,
            )
            if (
                supplied.resolve() != expected
                or file_sha256(supplied) != result["continuous_ego_video_sha256"][key]
            ):
                raise ValueError(
                    "Policy ego must use the original conditioned sensor recording and exact frame clock"
                )
        policy_ego_evidence = {
            "source_resolution": [640, 480],
            "condition": profile["subtype"],
            "effective_sensor_resolution": [
                round(640 * REDUCED_RESOLUTION_SCALE),
                round(480 * REDUCED_RESOLUTION_SCALE),
            ]
            if profile["subtype"] == "reduced_resolution"
            else [640, 480],
            "low_light_exposure_scale": LOW_LIGHT_EXPOSURE_SCALE
            if profile["subtype"] == "low_light"
            else None,
            "source": "original post-condition robot sensor stream; also consumed by policy and mapping",
            "continuous_video_sha256": result["continuous_ego_video_sha256"]["ego"],
            "frame_manifest_sha256": result["continuous_ego_video_sha256"][
                "frame_manifest"
            ],
            "additional_model_calls": 0,
        }
    if args.ego_frontier_cache:
        cache = json.loads(args.ego_frontier_cache.read_bytes())
        archived = cache["presentation"]
        if file_sha256(archived["path"]) != archived["sha256"]:
            raise ValueError("Frontier cache presentation hash mismatch")
        previous = json.loads(Path(archived["path"]).read_bytes())
        if (
            previous["source_result_sha256"] != hashlib.sha256(result_bytes).hexdigest()
            or previous["source_heartbeats_sha256"]
            != hashlib.sha256(heartbeat_bytes).hexdigest()
            or previous["source_rollout"] != source_rollout
        ):
            raise ValueError(
                "Frontier cache differs from the immutable result/rollout/heartbeats"
            )
    annotations, annotation_receipt = None, None
    if args.annotations:
        annotations, annotation_receipt = load_editorial_annotations(
            path=args.annotations,
            result_path=args.result,
            result=result,
            rollout=rollout,
            heartbeats=beats,
            heartbeats_path=args.heartbeats,
            source_result_root=args.source_result_root,
        )
    models = list(result.get("agent_model_usage", {}).get("by_model", {}))
    label = " / ".join(models) or str(result.get("model") or "")
    model_observations, observation_rollout = None, None
    grounding_depth_observations, grounding_depth_rollout = None, None
    if args.show_model_observation:
        model_observations, observation_rollout = load_model_observations(
            result=result,
            result_dir=args.result.parent,
            model_input_dir=args.model_input_dir,
            **(
                {"source_result_root": args.source_result_root}
                if args.source_result_root
                else {}
            ),
        )
    if args.show_grounding_depth:
        grounding_depth_observations, grounding_depth_rollout = (
            load_conditioned_depth_observations(
                result=result,
                result_dir=args.result.parent,
                **(
                    {"source_result_root": args.source_result_root}
                    if args.source_result_root
                    else {}
                ),
            )
        )
    lidar_observations = None
    if args.lidar_input_dir:
        from tarsphere.infra.perception.presentation_radar import (
            load_lidar_observations,
        )

        lidar_observations = load_lidar_observations(
            rollout=rollout,
            model_input_dir=args.lidar_input_dir,
            result_dir=args.result.parent,
        )
    frontier_model_source = None
    if args.ego_frontier_labels:
        continuous = result["continuous_ego_videos"]
        frontier_model_source = {}
        for key in ("ego", "frame_manifest"):
            path = _recorded_artifact_path(
                continuous[key],
                result_dir=args.result.parent,
                source_result_root=args.source_result_root,
            )
            digest = file_sha256(path)
            if digest != result["continuous_ego_video_sha256"][key]:
                raise ValueError(
                    "Ego label seed recording differs from its immutable source hash"
                )
            frontier_model_source[key] = str(path)
            frontier_model_source[key + "_sha256"] = digest
    metadata = render_demo_presentation(
        output_path=args.output,
        ego_path=args.ego,
        ego_manifest=args.ego_frames,
        observer_path=args.observer,
        observer_manifest=args.observer_frames,
        trace=trace,
        heartbeats=beats,
        online_map_dir=args.online_map_dir,
        human_replay_log=human_replay_log,
        task_title=args.task_title,
        source_label=label,
        height=args.height,
        seconds=args.seconds,
        archive_preview=args.archive_layout_preview,
        ffmpeg=args.ffmpeg,
        single_ego=single_ego,
        full_hud=args.ego_full_hud,
        display_grade=args.display_grade,
        debug_preview=args.debug_640_preview,
        policy_ego_evidence=policy_ego_evidence,
        preserve_map=args.preserve_map,
        ego_frontier_labels=args.ego_frontier_labels,
        frontier_model_source=frontier_model_source,
        ego_frontier_cache=args.ego_frontier_cache,
        display_affine_manifest=args.display_affine_manifest,
        frontier_captures=[
            dict(
                action_step=e["decision_step"],
                observation_step=e["observation"]["step"],
                physics_step=e["observation"]["camera_capture"]["physics_step"],
                capture_id=e["observation"]["camera_capture"]["capture_id"],
            )
            for e in rollout["evaluation_steps"]
            if e.get("observation", {}).get("camera_capture")
        ]
        if args.ego_frontier_labels
        else None,
        terminal_hold_seconds=args.terminal_hold_seconds,
        lidar_observations=lidar_observations,
        recorded_result={
            **{
                key: result.get(key)
                for key in ("success", "outcome_status", "termination_reason")
            },
            "safety_violations": [
                {"type": violation["type"]}
                for violation in (result.get("safety_violations") or [])
                if isinstance(violation, dict) and "type" in violation
            ],
        },
        show_model_observation=args.show_model_observation,
        model_observations=model_observations,
        show_grounding_depth=args.show_grounding_depth,
        grounding_depth_observations=grounding_depth_observations,
        **({"annotations": annotations} if annotations else {}),
    )
    metadata.update(
        source_result=str(args.result.resolve()),
        source_heartbeats=str(args.heartbeats.resolve()),
        source_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        source_heartbeats_sha256=hashlib.sha256(heartbeat_bytes).hexdigest(),
        source_rollout=source_rollout,
    )
    if args.source_result_root:
        metadata["source_result_mapping"] = {
            "recorded_root": str(args.source_result_root),
            "local_root": str(args.result.parent.resolve()),
        }
    if annotation_receipt:
        metadata["editorial_annotations"] = annotation_receipt
    if observation_rollout:
        metadata["model_observation_rollout"] = observation_rollout
    if grounding_depth_rollout:
        metadata["grounding_depth_rollout"] = grounding_depth_rollout
    args.output.with_suffix(".presentation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "video": metadata["video"],
                "frame_count": metadata["frame_count"],
                "synchronization_verified": metadata["synchronization_verified"],
            }
        )
    )


if __name__ == "__main__":
    main()
