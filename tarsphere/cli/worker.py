#!/usr/bin/env python
"""Persistent Isaac worker: cold-load each scene once, soft-reset its episodes.

Pass multiple --episode arguments to reuse the same environment for nav/fetch.
Scene USD, materials and camera infrastructure remain resident; episode actors,
targets and robot state reset through load_episode + reset. Harness reference
always disables NavMesh and has no execution/physics-step budget. Its metric is
measured physics steps, while trace indices only identify evidence frames.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# A worker is launched by file path, so Python otherwise puts ``tarsphere/cli/``—not
# the repository root—at sys.path[0].  Pin the owning checkout before importing
# tarsphere; an installed or sibling checkout must never supply half the
# runtime contract.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
from tarsphere.infra.process_lifecycle import require_parent_lifetime
from tarsphere.evaluation.result_layout import (
    result_directory,
    prepare_result_directory,
)

require_parent_lifetime("TARSPHERE_SCHEDULER_PID")

# ── GPU bootstrap (MUST run before any torch / Isaac import) ──────────────
# Isaac Sim's Kit runtime must own the CUDA context first; if torch (pulled in
# by SAM3 during agent construction) initialises CUDA before Isaac,
# the two contexts contend and the machine can freeze on the OOM path.
from tarsphere.infra.simulation.gpu_bootstrap import configure_cuda_allocator

configure_cuda_allocator()

from tarsphere.configs.secrets import DEFAULT_AGENT_MAX_TOKENS
from tarsphere.harness.factory import build_robot_agent_envelope
from tarsphere.harness.harness import AgentHarnessConfig
from tarsphere.harness.harness_cli import (
    add_agent_harness_arguments,
    agent_harness_config_from_args,
)
from tarsphere.harness.warmup import (
    heavy_backend_warmup_requested,
    warmup_configured_heavy_backends,
)
from tarsphere.api_usage import aggregate_model_calls, merge_usage_reports
from tarsphere.bench.case_loader import BenchmarkCaseLoader
from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.configs.secrets import EvaluationSupportServiceError
from tarsphere.evaluation.official import (
    FORMAL_EVALUATION_MAX_DECISION_STEPS,
    validate_formal_episode_runtime,
)
from tarsphere.evaluation.online_scoring import score_saved_result
from tarsphere.infra.control.physical_contracts import SIMULATION_PHYSICS_HZ
from tarsphere.infra.episode.episode_runner import AutonomousEpisodeRunner
from tarsphere.infra.episode.errors import EpisodeInitializationError
from tarsphere.infra.episode.human_actor_runtime import HumanActorRuntime
from tarsphere.infra.episode.rollout_control import ProgressWatchdog
from tarsphere.infra.perception.continuous_ego_video import (
    ContinuousEgoVideoArtifacts,
)
from tarsphere.infra.perception.room_observer import (
    ROOM_OBSERVER_VIEW_NAMES,
)
from tarsphere.infra.simulation.isaac_env import (
    VIEWER_MODES,
    IsaacAutonomousEnv,
    RenderMode,
    RobotPoseUnavailableError,
)
from tarsphere.infra.simulation.physical_scene_preparation import (
    PhysicalScenePreparationError,
)
from tarsphere.infra.simulation.scene_server_preflight import (
    preload_open_frontier_clustering_dependencies,
)

_KEEPALIVE_REFS: list[Any] = []


class EpisodeInfrastructureQuarantined(RuntimeError):
    """The resident simulator is unsafe to reuse for another official row."""


def _requires_simulator_quarantine(exc: BaseException) -> bool:
    """Return whether an invalid attempt may have poisoned resident Isaac."""

    current: BaseException | None = exc
    visited: set[int] = set()
    infrastructure_types = (
        EpisodeInitializationError,
        PhysicalScenePreparationError,
        RobotPoseUnavailableError,
        MemoryError,
        SystemError,
    )
    while current is not None and id(current) not in visited:
        if isinstance(current, infrastructure_types):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _is_evaluation_support_outage(exc: BaseException | None) -> bool:
    """Identify a bounded support-API failure without hiding other defects."""

    current = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, EvaluationSupportServiceError):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


# ── Grouping ──────────────────────────────────────────────────────────────


def _group_key(episode) -> str:
    """Episodes that can share one loaded environment: same scene.

    Robot start is reset per episode and therefore does not split a scene group.
    """
    return episode.scene.asset_path


def group_episodes_by_scene(episodes: list) -> list[list]:
    """Group episodes preserving first-seen order.

    Episodes sharing a scene reuse one environment. Only a scene change starts
    a new group. NavMesh baking is independently disabled for harness reference.
    """
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for ep in episodes:
        groups.setdefault(_group_key(ep), []).append(ep)
    return list(groups.values())


# ── Output helpers ────────────────────────────────────────────────────────


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "episode"


def _preload_open_frontier_clustering_if_needed(
    agent_harness_config: AgentHarnessConfig,
) -> None:
    """Import HDBSCAN before SimulationApp prepends Kit's vendored SciPy."""

    dependency_paths = preload_open_frontier_clustering_dependencies()
    print(
        "[WORKER] OpenFrontier clustering dependencies preloaded before Isaac: "
        f"{dependency_paths}",
        flush=True,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".json.pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recording_artifact_requirements(
    result: dict, observer_views: tuple[str, ...]
) -> list[tuple[str, bool]]:
    """Require recorded sources here; shared demo panels are composed afterwards."""
    video = result.get("video") or {}
    required = [
        (
            "raw_ego_video",
            all(
                video.get(key)
                for key in (
                    "path",
                    "sha256",
                    "manifest_path",
                    "manifest_sha256",
                )
            ),
        )
    ]
    if "presentation_ego" in observer_views:
        observers = result.get("observer_videos") or {}
        for view in observer_views:
            source = observers.get(view) or {}
            required.append(
                (
                    f"observer_{view}",
                    bool(
                        source.get("path")
                        and source.get("frame_manifest")
                        and source.get("frame_count", 0) > 0
                    ),
                )
            )
    else:
        videos = result.get("continuous_ego_videos") or {}
        required.extend(
            (name, bool(videos.get(name)))
            for name in (
                "ego_decision_panel",
                "online_map",
                "model_output_ass",
            )
        )
    return required


def _save_result(result: dict, output_dir: Path) -> Path:
    filename = (
        "result.json"
        if "evaluation_identity" in result
        else (f"{_safe_filename(result.get('episode_id', 'episode'))}.json")
    )
    path = output_dir / filename
    _atomic_json(path, result)
    return path


def _next_physical_attempt(output_dir: Path, episode_id: str) -> tuple[int, Path]:
    artifact_dir = output_dir / "artifacts" / _safe_filename(episode_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    used = []
    for path in artifact_dir.glob("physical_attempt_*.rollout.json"):
        match = re.fullmatch(r"physical_attempt_(\d+)\.rollout\.json", path.name)
        if match:
            used.append(int(match.group(1)))
    return max(used, default=0) + 1, artifact_dir


def _save_rollout_artifact(
    episode: TARSphereEpisode,
    runner: AutonomousEpisodeRunner | None,
    result: dict[str, Any],
    output_dir: Path,
) -> tuple[int | None, Path, str]:
    """Persist every physical attempt before any offline evaluator is called."""

    formal = "evaluation_identity" in result
    if formal:
        attempt = None
        path = output_dir / "rollout.json"
        episode_path = output_dir / "episode.json"
    else:
        attempt, artifact_dir = _next_physical_attempt(
            output_dir, str(result["episode_id"])
        )
        path = artifact_dir / f"physical_attempt_{attempt:03d}.rollout.json"
        episode_path = artifact_dir / f"physical_attempt_{attempt:03d}.episode.json"
    _atomic_json(episode_path, episode.model_dump(mode="json"))
    episode_digest = _sha256(episode_path)
    evaluation_steps = [
        step.model_dump(mode="json")
        for step in getattr(runner, "_evaluation_steps", ())
        if runner is not None
    ]
    artifact = {
        "schema_version": "tarsphere.rollout_artifact.v1",
        "episode_id": result["episode_id"],
        **({"physical_attempt": attempt} if not formal else {}),
        "episode_artifact_path": str(episode_path),
        "episode_artifact_sha256": episode_digest,
        "termination_reason": result.get("termination_reason"),
        "decision_steps": result.get("decision_steps"),
        "action_steps": result.get("action_steps"),
        "physics_steps": result.get("physics_steps"),
        "trace": list(result.get("trace") or ()),
        "human_replay_log": list(result.get("human_replay_log") or ()),
        **(
            {"conditioned_depth_captures": result["conditioned_depth_captures"]}
            if result.get("conditioned_depth_captures")
            else {}
        ),
        "heartbeats": list(getattr(runner, "heartbeats", ()))
        if runner is not None
        else [],
        "evaluation_steps": evaluation_steps,
        "heartbeat_path": result.get("heartbeat_path"),
        "heartbeat_sha256": result.get("heartbeat_sha256"),
        "model_output_trace_path": result.get("model_output_trace_path"),
        "model_output_trace_sha256": result.get("model_output_trace_sha256"),
        "video": result.get("video"),
        "continuous_ego_videos": result.get("continuous_ego_videos"),
        "continuous_ego_video_sha256": result.get("continuous_ego_video_sha256"),
        "video_recording_enabled": result.get("video_recording_enabled", False),
        "failure": result.get("failure"),
    }
    _atomic_json(path, artifact)
    digest = _sha256(path)
    if not formal:
        result["physical_attempt"] = attempt
    result["episode_artifact_path"] = str(episode_path)
    result["episode_artifact_sha256"] = episode_digest
    result["rollout_artifact_path"] = str(path)
    result["rollout_artifact_sha256"] = digest
    return attempt, path, digest


def _save_trajectory(
    runner: AutonomousEpisodeRunner,
    output_dir: Path,
    *,
    physical_attempt: int | None,
) -> tuple[Path, str]:
    """Persist the immutable typed trajectory before any evaluator runs."""

    trajectory = runner.evaluation_trajectory()
    artifact_dir = output_dir / "artifacts" / _safe_filename(trajectory.episode_id)
    path = (
        output_dir / "evaluation_trajectory.json"
        if physical_attempt is None
        else artifact_dir
        / (f"physical_attempt_{physical_attempt:03d}.evaluation_trajectory.json")
    )
    if path.exists():
        raise FileExistsError(f"evaluation trajectory already exists: {path}")
    _atomic_json(path, trajectory.model_dump(mode="json"))
    return path, _sha256(path)


def _defer_saved_trajectory(
    *,
    runner: AutonomousEpisodeRunner,
    result: dict[str, Any],
    output_dir: Path,
    physical_attempt: int | None,
) -> None:
    """Persist a frozen trajectory for the separately launched evaluator."""

    trajectory_path, trajectory_sha256 = _save_trajectory(
        runner, output_dir, physical_attempt=physical_attempt
    )
    result["evaluation_trajectory_path"] = str(trajectory_path)
    result["evaluation_trajectory_sha256"] = trajectory_sha256
    result["evaluation_attempts"] = 0
    result["success"] = False
    result["outcome_status"] = "pending_evaluation"
    _save_result(result, output_dir)

    result["api_usage"] = merge_usage_reports(
        (
            result.get("agent_model_usage") or aggregate_model_calls([]),
            result.get("human_model_usage") or aggregate_model_calls([]),
            result.get("evaluation_model_usage") or aggregate_model_calls([]),
        )
    )


def _render_ego_decision_panel(
    *,
    output_dir: Path,
    slug: str,
    artifacts: ContinuousEgoVideoArtifacts,
    decision_trace: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    heartbeats: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, str]:
    """Compose the original ego/map overlay with refined dark styling."""
    from tarsphere.infra.perception.ego_presentation import render_ego_presentation

    return render_ego_presentation(
        output_dir=output_dir,
        slug=slug,
        artifacts=artifacts,
        decision_trace=decision_trace,
        heartbeats=heartbeats,
    )


def _summary_line(episode, result: dict) -> str:
    steps = (
        ""
        if "reference_status" in result
        else f"steps={result.get('decision_steps')}  "
    )
    return (
        f"[WORKER] {result.get('episode_id', '?')}: "
        f"success={result.get('success')}  "
        f"{steps}"
        f"physics={result.get('physics_steps', '?')}  "
        f"renders={result.get('render_count', '?')}  "
        f"mode={result.get('render_mode', '?')}  "
        f"wall={result.get('wall_clock_s', '?')}s"
    )


# ── Core run ──────────────────────────────────────────────────────────────


def _create_batch_environment(
    episode: TARSphereEpisode,
    *,
    headless: bool,
    image_dir: str,
    viewer_mode: str,
    room_observer_view_names: tuple[str, ...],
    room_observer_resolution: tuple[int, int],
    room_observer_stride: int,
    room_observer_max_frames: int | None,
    agent_harness_config: AgentHarnessConfig,
    render_mode: RenderMode | str = RenderMode.EVALUATION,
    human_motion_backend: str = "kinematic",
) -> IsaacAutonomousEnv:
    """Create an offline worker environment without a WebRTC render pipeline."""
    return IsaacAutonomousEnv(
        episode,
        headless=headless,
        image_dir=image_dir,
        viewer_mode=viewer_mode,
        enable_streaming=False,
        enable_human_ego_cameras=False,
        room_observer_view_names=room_observer_view_names,
        room_observer_resolution=room_observer_resolution,
        room_observer_stride=room_observer_stride,
        room_observer_max_frames=room_observer_max_frames,
        agent_harness_config=agent_harness_config,
        render_mode=render_mode,
        human_motion_backend=human_motion_backend,
    )


def _run_group(
    episodes: list,
    *,
    agent: Any,
    headless: bool,
    image_dir: str,
    viewer_mode: str,
    max_steps: int | None,
    output_dir: Path,
    prebuilt_env: IsaacAutonomousEnv | None = None,
    prebuilt_initial_observation: dict | None = None,
    keep_env_alive: bool = False,
    defer_env_close: bool = False,
    skip_navmesh: bool = False,
    room_observer_view_names: tuple[str, ...] = (),
    room_observer_resolution: tuple[int, int] = (640, 360),
    room_observer_stride: int = 1,
    room_observer_max_frames: int | None = None,
    agent_harness_config: AgentHarnessConfig = AgentHarnessConfig(),
    render_mode: RenderMode | str = RenderMode.EVALUATION,
    human_motion_backend: str = "kinematic",
    evaluation_identity_by_episode: dict[str, dict[str, str]] | None = None,
    evaluate_after_rollout: bool = False,
    record_video: bool = False,
) -> list[dict]:
    """Run one loaded scene, then reset each episode via load_episode.

    Robot, NPC and target state reset between tasks without reloading the scene.

    ``prebuilt_env`` lets the caller pass an already-created (and reset) env —
    used for the first group, which is primed before the agent is built so that
    Isaac owns the CUDA context. When provided, this function takes ownership
    of closing it.
    """
    print(
        f"[WORKER] Scene group: {len(episodes)} episode(s), scene="
        f"{episodes[0].scene.asset_path}",
        flush=True,
    )

    local_initial_observation: dict | None = None
    if prebuilt_env is not None:
        env = prebuilt_env
    else:
        env = _create_batch_environment(
            episodes[0],
            headless=headless,
            image_dir=image_dir,
            viewer_mode=viewer_mode,
            room_observer_view_names=room_observer_view_names,
            room_observer_resolution=room_observer_resolution,
            room_observer_stride=room_observer_stride,
            room_observer_max_frames=room_observer_max_frames,
            agent_harness_config=agent_harness_config,
            render_mode=render_mode,
            human_motion_backend=human_motion_backend,
        )
        local_initial_observation = env.reset(skip_navmesh=skip_navmesh)

    results: list[dict] = []
    score_executor = (
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="online-score")
        if evaluate_after_rollout
        else None
    )
    score_futures: list[Future[dict[str, Any]]] = []
    batch_output_dir = output_dir
    try:
        for idx, episode in enumerate(episodes):
            t0 = time.monotonic()
            if evaluation_identity_by_episode is not None:
                profile = episode.evaluation.task_profile
                output_dir = result_directory(
                    batch_output_dir,
                    profile.mainline.value,
                    profile.capability_slice.value,
                )
                prepare_result_directory(output_dir)
                expected_physical_attempt = None
                artifact_slug = "episode"
            else:
                expected_physical_attempt, _ = _next_physical_attempt(
                    output_dir, episode.episode_id
                )
                artifact_slug = (
                    f"{_safe_filename(episode.episode_id)}_"
                    f"physical_attempt_{expected_physical_attempt:03d}"
                )
            runner: AutonomousEpisodeRunner | None = None
            observer_frame_root: Path | None = None
            continuous_ego_artifacts: ContinuousEgoVideoArtifacts | None = None
            heartbeat_path = output_dir / f"{artifact_slug}.heartbeat.jsonl"
            model_output_path = output_dir / f"{artifact_slug}.model_output.jsonl"
            infrastructure_error: BaseException | None = None
            quarantine_error: BaseException | None = None
            defer_remaining_episodes = False
            try:
                if record_video:
                    online_map_dir = output_dir / "online_maps" / artifact_slug
                    if online_map_dir.exists():
                        # A process can die before its immutable rollout artifact
                        # is committed. The same physical-attempt number then owns
                        # these incomplete presentation files on resume.
                        shutil.rmtree(online_map_dir)
                    online_map_dir.mkdir(parents=True, exist_ok=False)
                    os.environ["TARSPHERE_RECORD_DIR"] = str(online_map_dir)
                    os.environ["TARSPHERE_NAVIGATION_DEBUG_ARTIFACTS"] = "1"
                else:
                    os.environ.pop("TARSPHERE_RECORD_DIR", None)
                    os.environ["TARSPHERE_NAVIGATION_DEBUG_ARTIFACTS"] = "0"
                env.configure_continuous_ego_video(
                    enabled=record_video,
                    output_dir=output_dir / "videos",
                    slug=artifact_slug,
                    capture_stride=8,
                    include_bev=False,
                )
                if room_observer_view_names:
                    observer_frame_root = (
                        output_dir / ".observer_frames" / artifact_slug
                    )
                    if observer_frame_root.exists():
                        shutil.rmtree(observer_frame_root)
                    env.image_dir = observer_frame_root
                initial_observation = (
                    (
                        prebuilt_initial_observation
                        if prebuilt_env is not None
                        else local_initial_observation
                    )
                    if idx == 0
                    else None
                )
                if initial_observation is None:
                    # Same scene: reset task state without another scene load.
                    # Reset owns start_episode; open video streams only after
                    # it finishes so a warm episode cannot reset live encoders.
                    env.load_episode(episode)
                    try:
                        initial_observation = env.reset(skip_navmesh=skip_navmesh)
                    except Exception as exc:
                        raise EpisodeInitializationError(
                            f"Episode {episode.episode_id!r} initialization failed: {exc}"
                        ) from exc
                else:
                    # The priming reset preceded per-episode recorder setup.
                    env._start_continuous_ego_video()
                    if observer_frame_root is not None:
                        env._start_room_observer_episode()
                if observer_frame_root is not None:
                    env._room_observer.start_video(output_dir / "videos", artifact_slug)
                    env.capture_initial_presentation_frame()
                runner = AutonomousEpisodeRunner(
                    episode=episode,
                    environment=env,
                    agent=agent,
                    human=HumanActorRuntime(
                        episode,
                    ),
                    heartbeat_path=heartbeat_path,
                    model_output_path=model_output_path,
                    progress_watchdog=ProgressWatchdog(),
                    agent_harness_config=agent_harness_config,
                )
                result = runner.run(
                    max_decision_steps=max_steps,
                    initial_observation=initial_observation,
                )
            except Exception as exc:  # noqa: BLE001
                # Every model-caused outcome is represented as a normal
                # decision/termination record. An exception escaping the
                # runner is therefore invalid harness or infrastructure, never
                # an Agent failure eligible for scoring.
                infrastructure_error = exc
                if _requires_simulator_quarantine(exc):
                    quarantine_error = exc
                try:
                    result = (
                        runner.partial_result(failure=exc)
                        if runner is not None
                        else {
                            "episode_id": episode.episode_id,
                            "success": False,
                        }
                    )
                except EpisodeInitializationError:
                    # partial_result intentionally re-raises initialization
                    # failures for interactive callers. The batch worker still
                    # persists the empty/partial attempt before quarantining
                    # this resident simulator.
                    result = {
                        "episode_id": episode.episode_id,
                        "success": False,
                        "outcome_status": "infrastructure_invalid",
                        "termination_reason": "initialization_failed",
                        "decision_steps": 0,
                        "action_steps": len(getattr(runner, "trace", ())),
                        "physics_steps": getattr(env, "physics_steps", None),
                        "render_count": getattr(env, "render_count", None),
                        "trace": list(getattr(runner, "trace", ())),
                        "human_replay_log": (
                            runner._human_replay_log() if runner is not None else []
                        ),
                        "safety_violations": list(
                            getattr(runner, "safety_violations", ())
                        ),
                        "agent_model_usage": aggregate_model_calls([]),
                        "human_model_usage": aggregate_model_calls([]),
                    }
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc)
                result["traceback"] = traceback.format_exc()
                result["wall_clock_s"] = round(time.monotonic() - t0, 1)
                if infrastructure_error is not None:
                    result["success"] = False
                    result["outcome_status"] = "infrastructure_invalid"
                    result["infrastructure_error"] = {
                        "type": type(infrastructure_error).__name__,
                        "message": str(infrastructure_error),
                    }
                print(
                    f"[WORKER] {episode.episode_id}: failed with "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                if record_video:
                    try:
                        continuous_ego_artifacts = env.finish_continuous_ego_video()
                        if continuous_ego_artifacts is None:
                            raise RuntimeError(
                                "Episode produced no continuous ego recording."
                            )
                        result["video"] = {
                            "path": str(continuous_ego_artifacts.ego_video_path),
                            "sha256": _sha256(continuous_ego_artifacts.ego_video_path),
                            "view": "ego",
                            "manifest_path": str(
                                continuous_ego_artifacts.manifest_path
                            ),
                            "manifest_sha256": _sha256(
                                continuous_ego_artifacts.manifest_path
                            ),
                            "resolution": list(continuous_ego_artifacts.resolution),
                            "fps": continuous_ego_artifacts.fps,
                            "frame_count": len(continuous_ego_artifacts.physics_steps),
                        }
                        if "presentation_ego" in room_observer_view_names:
                            # The demo launcher composes both clean views after
                            # observers finish. Keep the model camera as evidence.
                            continuous_videos = {
                                "ego": str(continuous_ego_artifacts.ego_video_path),
                                "frame_manifest": str(
                                    continuous_ego_artifacts.manifest_path
                                ),
                            }
                        else:
                            continuous_videos = _render_ego_decision_panel(
                                output_dir=output_dir,
                                slug=artifact_slug,
                                artifacts=continuous_ego_artifacts,
                                decision_trace=result.get("trace") or (),
                                heartbeats=(
                                    runner.heartbeats if runner is not None else ()
                                ),
                            )
                        result["continuous_ego_videos"] = continuous_videos
                        result["continuous_ego_video_sha256"] = {
                            name: _sha256(Path(video_path))
                            for name, video_path in continuous_videos.items()
                        }
                        result["video_path"] = continuous_videos[
                            "ego"
                            if "presentation_ego" in room_observer_view_names
                            else "ego_decision_panel"
                        ]
                    except Exception as exc:
                        result["video_error"] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        if episode.evaluation is not None:
                            if infrastructure_error is None:
                                infrastructure_error = exc
                            result["success"] = False
                            result["outcome_status"] = "infrastructure_invalid"
                            result["infrastructure_error"] = {
                                "type": type(exc).__name__,
                                "message": (
                                    "Required requested ego/decision-panel artifact "
                                    f"failed: {exc}"
                                ),
                            }
                            defer_remaining_episodes = True
                if observer_frame_root is not None:
                    try:
                        profile = getattr(
                            getattr(episode, "evaluation", None), "task_profile", None
                        )
                        if (
                            getattr(profile, "subtype", None) == "depth_noise"
                            and "presentation_ego" in room_observer_view_names
                            and runner is not None
                        ):
                            from tarsphere.infra.perception.demo_sensor_evidence import (
                                preserve_conditioned_depth_captures,
                            )

                            result["conditioned_depth_captures"] = (
                                preserve_conditioned_depth_captures(
                                    runner._evaluation_steps,
                                    output_dir=output_dir
                                    / "sensor_evidence"
                                    / artifact_slug,
                                    capture_root=observer_frame_root,
                                )
                            )
                        result["observer_videos"] = env._room_observer.finish_video()
                        if set(result["observer_videos"]) != set(
                            room_observer_view_names
                        ):
                            raise RuntimeError(
                                "Not every requested observer produced a recording."
                            )
                        if observer_frame_root.exists():
                            shutil.rmtree(observer_frame_root)
                    except Exception as exc:
                        result["observer_video_error"] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
            if heartbeat_path.is_file():
                result["heartbeat_path"] = str(heartbeat_path)
                result["heartbeat_sha256"] = _sha256(heartbeat_path)
            if model_output_path.is_file():
                result["model_output_trace_path"] = str(model_output_path)
                result["model_output_trace_sha256"] = _sha256(model_output_path)
            result["video_recording_enabled"] = bool(record_video)
            if episode.evaluation is not None:
                required_artifacts = [
                    (
                        "heartbeat",
                        bool(
                            result.get("heartbeat_path")
                            and result.get("heartbeat_sha256")
                        ),
                    ),
                    (
                        "model_output_trace",
                        bool(
                            result.get("model_output_trace_path")
                            and result.get("model_output_trace_sha256")
                        ),
                    ),
                ]
                if record_video:
                    required_artifacts.extend(
                        _recording_artifact_requirements(
                            result,
                            room_observer_view_names,
                        )
                    )
                missing_artifacts = [
                    name for name, present in required_artifacts if not present
                ]
                if missing_artifacts:
                    artifact_error = RuntimeError(
                        "Formal attempt is missing required forensic "
                        f"artifacts: {', '.join(missing_artifacts)}"
                    )
                    if infrastructure_error is None:
                        infrastructure_error = artifact_error
                    result["success"] = False
                    result["outcome_status"] = "infrastructure_invalid"
                    result["infrastructure_error"] = {
                        "type": "RequiredFormalArtifactError",
                        "message": str(artifact_error),
                    }
                    defer_remaining_episodes = True
            if result.get("termination_reason") in {
                "model_service_retry_exhausted",
                "model_request_failed",
            }:
                transport_exhausted = result.get("termination_reason") == (
                    "model_service_retry_exhausted"
                )
                result["success"] = False
                result["outcome_status"] = "infrastructure_invalid"
                result["infrastructure_error"] = {
                    "type": (
                        "AgentModelServiceError"
                        if transport_exhausted
                        else "AgentModelRequestError"
                    ),
                    "message": (
                        "The evaluated Agent endpoint exhausted transport "
                        "recovery; this physical attempt is not a model outcome."
                        if transport_exhausted
                        else "The evaluated Agent request failed before a "
                        "provider response; check the model binding and request "
                        "contract. This is not a model outcome."
                    ),
                }
                defer_remaining_episodes = True
            elif _is_evaluation_support_outage(infrastructure_error):
                result["termination_reason"] = (
                    "evaluation_support_service_retry_exhausted"
                )
                defer_remaining_episodes = True
            result["wall_clock_s"] = round(time.monotonic() - t0, 1)
            result["human_motion_backend"] = human_motion_backend
            result["agent_harness_config"] = agent_harness_config.as_dict()
            if getattr(agent, "certification_only", False):
                result["diagnostic_only"] = True
            if evaluation_identity_by_episode is not None:
                identity = evaluation_identity_by_episode.get(episode.episode_id)
                if identity is None:
                    raise RuntimeError(
                        f"Missing formal result identity for {episode.episode_id!r}"
                    )
                result["evaluation_identity"] = dict(identity)
            from tarsphere.bench.execution_metrics import attach_execution_quality

            attach_execution_quality(
                result,
                episode=episode,
                environment=env if runner is not None else None,
            )
            physical_attempt, _, _ = _save_rollout_artifact(
                episode, runner, result, output_dir
            )
            if physical_attempt != expected_physical_attempt:
                raise RuntimeError(
                    f"{episode.episode_id}: physical attempt changed from "
                    f"{expected_physical_attempt} to {physical_attempt}"
                )
            # Establish a durable pending/failed record before any networked
            # Judge call. A worker crash during evaluation must never erase the
            # completed physical rollout or make the scheduler rerun physics.
            _save_result(result, output_dir)
            if (
                episode.evaluation is not None
                and runner is not None
                and infrastructure_error is None
                and result.get("termination_reason")
                not in {
                    "model_service_retry_exhausted",
                    "model_request_failed",
                }
                and getattr(agent, "name", None)
                not in {"golden_oracle", "harness_reference"}
            ):
                try:
                    _defer_saved_trajectory(
                        runner=runner,
                        result=result,
                        output_dir=output_dir,
                        physical_attempt=physical_attempt,
                    )
                except ValueError as exc:
                    # No typed step means no valid physical trajectory exists.
                    result["success"] = False
                    result["outcome_status"] = "infrastructure_invalid"
                    result["infrastructure_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            if getattr(agent, "name", None) == "harness_reference":
                agent.finalize(runner, result)
            results.append(result)
            result_path = _save_result(result, output_dir)
            attempt_result_path = None
            if physical_attempt is not None:
                attempt_result_path = (
                    output_dir
                    / "artifacts"
                    / _safe_filename(episode.episode_id)
                    / f"physical_attempt_{physical_attempt:03d}.result.json"
                )
                _atomic_json(attempt_result_path, result)
            if (
                score_executor is not None
                and result.get("outcome_status") == "pending_evaluation"
            ):
                authoritative_result_path = attempt_result_path or result_path
                score_futures.append(
                    score_executor.submit(
                        score_saved_result,
                        authoritative_result_path=authoritative_result_path,
                        latest_result_path=(
                            result_path
                            if authoritative_result_path != result_path
                            else None
                        ),
                    )
                )
            print(_summary_line(episode, result), flush=True)
            if quarantine_error is not None:
                raise EpisodeInfrastructureQuarantined(
                    f"{episode.episode_id}: resident simulator quarantined "
                    f"after {type(quarantine_error).__name__}: "
                    f"{quarantine_error}"
                ) from quarantine_error
            if defer_remaining_episodes:
                remaining = len(episodes) - idx - 1
                print(
                    "[WORKER] Episode infrastructure recovery is required; "
                    f"deferring {remaining} remaining episode(s) in this "
                    "model batch while later model batches continue.",
                    flush=True,
                )
                break
            # Every episode is an independent official attempt. A policy,
            # gateway, or tool exception in one row must not suppress the
            # remaining rows for the already-loaded scene. The next iteration
            # performs the same complete Robot/NPC/target reset; a genuinely
            # broken simulator will therefore fail explicitly again.
    finally:
        if defer_env_close:
            print(
                "[WORKER] Returning the loaded scene to the schedule runner.",
                flush=True,
            )
        elif keep_env_alive:
            _KEEPALIVE_REFS.append(env)
            print(
                "[WORKER] Keeping Isaac environment alive for reuse "
                f"(env_ref_count={len(_KEEPALIVE_REFS)}).",
                flush=True,
            )
        else:
            env.close()
        if score_executor is not None:
            score_executor.shutdown(wait=True)
            # The scoring task persists its own typed failure result. Calling
            # result() here only surfaces an unexpected queue implementation
            # defect after every physical episode in this group has finished.
            for future in score_futures:
                future.result()
    return results


def _prime_isaac_context(
    episodes: list,
    *,
    headless: bool,
    image_dir: str,
    viewer_mode: str,
    skip_navmesh: bool = False,
    room_observer_view_names: tuple[str, ...] = (),
    room_observer_resolution: tuple[int, int] = (640, 360),
    room_observer_stride: int = 1,
    room_observer_max_frames: int | None = None,
    agent_harness_config: AgentHarnessConfig = AgentHarnessConfig(),
    render_mode: RenderMode | str = RenderMode.EVALUATION,
    human_motion_backend: str = "kinematic",
) -> tuple[IsaacAutonomousEnv, dict]:
    """Create the first scene's env and reset() so Isaac owns the CUDA context.

    The Kit/Isaac runtime must initialise CUDA before SAM3 (constructed
    inside the agent) lazily import torch — otherwise the contexts contend and
    the box can freeze on the OOM path. This creates the env for the first
    scene group and runs reset() once (which builds + bakes the scene), then
    returns the live env so the caller can reuse it for that group instead of
    paying for a second bake.

    The returned env has already reset()'d for its first episode; the caller's
    _run_group path must skip its own env construction for group 0.
    """
    print(
        f"[WORKER] Priming Isaac context for scene="
        f"{episodes[0].scene.asset_path} (before agent build)",
        flush=True,
    )
    env = _create_batch_environment(
        episodes[0],
        headless=headless,
        image_dir=image_dir,
        viewer_mode=viewer_mode,
        room_observer_view_names=room_observer_view_names,
        room_observer_resolution=room_observer_resolution,
        room_observer_stride=room_observer_stride,
        room_observer_max_frames=room_observer_max_frames,
        agent_harness_config=agent_harness_config,
        render_mode=render_mode,
        human_motion_backend=human_motion_backend,
    )
    observation = env.reset(skip_navmesh=skip_navmesh)
    return env, observation


def _build_worker_agent(
    *,
    api_key: str,
    fallback_api_keys: tuple[str, ...] = (),
    model: str,
    api_base: str,
    max_tokens: int = DEFAULT_AGENT_MAX_TOKENS,
    agent_harness_config: AgentHarnessConfig = AgentHarnessConfig(),
) -> Any:
    return build_robot_agent_envelope(
        api_key=api_key,
        fallback_api_keys=fallback_api_keys,
        model=model or None,
        api_base=api_base or None,
        max_tokens=max_tokens,
        agent_harness_config=agent_harness_config,
    )


def run_worker(
    episode_paths: list[str],
    *,
    api_key: str = "",
    model: str = "",
    api_base: str = "",
    max_tokens: int = DEFAULT_AGENT_MAX_TOKENS,
    headless: bool = False,
    image_dir: str = "",
    viewer_mode: str = "third_person",
    max_steps: int | None = None,
    output_dir: str = "",
    warmup_heavy_backends: bool = False,
    keep_alive: bool = False,
    skip_navmesh: bool = False,
    room_observer_view_names: tuple[str, ...] = (),
    room_observer_resolution: tuple[int, int] = (640, 360),
    room_observer_stride: int = 1,
    room_observer_max_frames: int | None = None,
    agent_harness_config: AgentHarnessConfig = AgentHarnessConfig(),
    render_mode: RenderMode | str = RenderMode.EVALUATION,
    human_motion_backend: str = "kinematic",
    evaluate_after_rollout: bool = False,
    record_video: bool = False,
) -> list[dict]:
    """Load, group, and run episodes in a persistent Isaac worker."""
    if not episode_paths:
        raise ValueError("At least one --episode is required.")
    if human_motion_backend not in {"kinematic", "animation"}:
        raise ValueError("human_motion_backend must be kinematic or animation")
    if human_motion_backend == "kinematic":
        skip_navmesh = True
    manager = BenchmarkCaseLoader()
    episodes = [manager.load_file(path) for path in episode_paths]

    groups = group_episodes_by_scene(episodes)
    print(
        f"[WORKER] {len(episodes)} episode(s) → {len(groups)} scene group(s) → "
        f"{0 if skip_navmesh else len(groups)} NavMesh bake(s), "
        f"{len(episodes) - len(groups)} same-scene soft reset(s).",
        flush=True,
    )
    _preload_open_frontier_clustering_if_needed(agent_harness_config)

    # Bring up Isaac FIRST: prime the first group's env + reset() so the Kit
    # runtime owns the CUDA context before the agent (and its SAM3
    # backends) lazily import torch. Building the agent before this used to race
    # Isaac for the context and could freeze the box on the OOM path.
    primed_env, primed_observation = _prime_isaac_context(
        groups[0],
        headless=headless,
        image_dir=image_dir,
        viewer_mode=viewer_mode,
        skip_navmesh=skip_navmesh,
        room_observer_view_names=room_observer_view_names,
        room_observer_resolution=room_observer_resolution,
        room_observer_stride=room_observer_stride,
        room_observer_max_frames=room_observer_max_frames,
        agent_harness_config=agent_harness_config,
        render_mode=render_mode,
        human_motion_backend=human_motion_backend,
    )

    agent = _build_worker_agent(
        api_key=api_key,
        model=model,
        api_base=api_base,
        max_tokens=max_tokens,
        agent_harness_config=agent_harness_config,
    )
    if warmup_heavy_backends or heavy_backend_warmup_requested():
        print(
            "[WORKER] Warming configured heavy backends after Isaac bootstrap.",
            flush=True,
        )
        warmup_summary = warmup_configured_heavy_backends(agent, primed_observation)
        print(f"[WORKER] Heavy backend warmup: {warmup_summary}", flush=True)

    out_path = Path(output_dir) if output_dir else None
    default_output_dir = Path.home() / ".local/share/tarsphere/worker_results"
    all_results: list[dict] = []
    for idx, group in enumerate(groups):
        # Group 0 reuses the env we just primed (no second bake); later groups
        # create + bake their own env inside _run_group as before.
        prebuilt = primed_env if idx == 0 else None
        prebuilt_observation = primed_observation if idx == 0 else None
        all_results.extend(
            _run_group(
                group,
                agent=agent,
                headless=headless,
                image_dir=image_dir,
                viewer_mode=viewer_mode,
                max_steps=max_steps,
                output_dir=out_path if out_path is not None else default_output_dir,
                prebuilt_env=prebuilt,
                prebuilt_initial_observation=prebuilt_observation,
                keep_env_alive=keep_alive and idx == len(groups) - 1,
                skip_navmesh=skip_navmesh,
                room_observer_view_names=room_observer_view_names,
                room_observer_resolution=room_observer_resolution,
                room_observer_stride=room_observer_stride,
                room_observer_max_frames=room_observer_max_frames,
                agent_harness_config=agent_harness_config,
                render_mode=render_mode,
                human_motion_backend=human_motion_backend,
                evaluate_after_rollout=evaluate_after_rollout,
                record_video=record_video,
            )
        )
    return all_results


def run_evaluation_schedule(
    schedule_path: Path,
    *,
    headless: bool,
    image_dir: str,
    viewer_mode: str,
    max_steps: int | None,
    warmup_heavy_backends: bool,
    skip_navmesh: bool,
    room_observer_view_names: tuple[str, ...],
    room_observer_resolution: tuple[int, int],
    room_observer_stride: int,
    room_observer_max_frames: int | None,
    agent_harness_config: AgentHarnessConfig,
    render_mode: RenderMode | str,
    human_motion_backend: str,
    evaluate_after_rollout: bool = False,
    record_video: bool = False,
) -> list[dict]:
    """Run one balanced worker lane containing one or two scene loads.

    Each scene stays resident for all of its scheduled model batches. The
    formal 45-worker plan gives primary lanes one scene and helper lanes two
    scenes, which is the minimum reload count compatible with five or six
    complete 12-episode model batches per worker.
    """
    raw = json.loads(schedule_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "tarsphere.isaac_worker_schedule.v4":
        raise ValueError("Unsupported evaluation schedule schema")
    scene_runs = list(raw.get("scene_runs", ()))
    if not scene_runs:
        raise ValueError("Evaluation schedule contains no pending scene runs")
    if human_motion_backend == "kinematic":
        skip_navmesh = True
    if max_steps is not None:
        raise ValueError("Formal evaluation schedule must not set --max-steps")

    _preload_open_frontier_clustering_if_needed(agent_harness_config)
    manager = BenchmarkCaseLoader()
    all_results: list[dict] = []
    for scene_index, scene_run in enumerate(scene_runs):
        expected_scene = str(scene_run["scene_asset_path"])
        scene_job_id = str(scene_run.get("scene_job_id") or "").strip()
        if not scene_job_id:
            raise ValueError("Formal schedule scene run is missing scene_job_id")
        loaded_runs: list[tuple[dict[str, Any], list[TARSphereEpisode]]] = []
        for model_run in scene_run.get("model_runs", ()):
            if int(model_run.get("max_tokens", DEFAULT_AGENT_MAX_TOKENS)) < 1:
                raise ValueError("Scheduled max_tokens must be positive")
            episode_rows = list(model_run.get("episodes", ()))
            if episode_rows:
                episodes: list[TARSphereEpisode] = []
                for row in episode_rows:
                    if not isinstance(row, dict) or set(row) != {
                        "episode_id",
                        "path",
                        "sha256",
                    }:
                        raise ValueError(
                            "Formal schedule episode rows require exactly "
                            "episode_id, path and sha256"
                        )
                    path = REPO_ROOT / str(row["path"])
                    actual_hash = _sha256(path)
                    if actual_hash != str(row["sha256"]):
                        raise ValueError(
                            f"Scheduled episode hash changed for {path}: "
                            f"schedule={row['sha256']} file={actual_hash}"
                        )
                    episode = manager.load_file(path)
                    if episode.episode_id != str(row["episode_id"]):
                        raise ValueError(
                            f"Scheduled episode ID differs from YAML for {path}: "
                            f"schedule={row['episode_id']!r} "
                            f"YAML={episode.episode_id!r}"
                        )
                    episodes.append(episode)
                loaded_runs.append(
                    (
                        model_run,
                        episodes,
                    )
                )
        if not loaded_runs:
            continue
        for _, episodes in loaded_runs:
            for episode in episodes:
                validate_formal_episode_runtime(episode)
        observed_scenes = {
            episode.scene.asset_path
            for _, episodes in loaded_runs
            for episode in episodes
        }
        if observed_scenes != {expected_scene}:
            raise ValueError(
                f"Scene schedule expected {expected_scene!r}, "
                f"got {sorted(observed_scenes)!r}"
            )

        print(
            f"[WORKER] Scene {scene_index + 1}/{len(scene_runs)}: "
            f"{expected_scene}; models={len(loaded_runs)}; episodes="
            f"{sum(len(episodes) for _, episodes in loaded_runs)}.",
            flush=True,
        )
        first_episodes = loaded_runs[0][1]
        env, initial_observation = _prime_isaac_context(
            first_episodes,
            headless=headless,
            image_dir=image_dir,
            viewer_mode=viewer_mode,
            skip_navmesh=skip_navmesh,
            room_observer_view_names=room_observer_view_names,
            room_observer_resolution=room_observer_resolution,
            room_observer_stride=room_observer_stride,
            room_observer_max_frames=room_observer_max_frames,
            agent_harness_config=agent_harness_config,
            render_mode=render_mode,
            human_motion_backend=human_motion_backend,
        )
        try:
            for model_index, (model_run, episodes) in enumerate(loaded_runs):
                api_key_env = str(model_run["api_key_env"])
                api_key = os.environ.get(api_key_env, "").strip()
                if not api_key:
                    raise ValueError(
                        f"{model_run['name']}: required API key variable "
                        f"{api_key_env} is empty"
                    )
                fallback_api_key_env = str(model_run.get("api_key_fallback_env") or "")
                fallback_api_key = (
                    os.environ.get(fallback_api_key_env, "").strip()
                    if fallback_api_key_env
                    else ""
                )
                fallback_api_key_2_env = str(
                    model_run.get("api_key_fallback_2_env") or ""
                )
                fallback_api_key_2 = (
                    os.environ.get(fallback_api_key_2_env, "").strip()
                    if fallback_api_key_2_env
                    else ""
                )
                print(
                    f"[WORKER] Model {model_index + 1}/{len(loaded_runs)}: "
                    f"{model_run['name']} ({len(episodes)} pending episodes).",
                    flush=True,
                )
                agent = _build_worker_agent(
                    api_key=api_key,
                    fallback_api_keys=tuple(
                        key for key in (fallback_api_key, fallback_api_key_2) if key
                    ),
                    model=str(model_run["model"]),
                    api_base=str(model_run["api_base"]),
                    max_tokens=int(
                        model_run.get("max_tokens", DEFAULT_AGENT_MAX_TOKENS)
                    ),
                    agent_harness_config=agent_harness_config,
                )
                if model_index == 0 and (
                    warmup_heavy_backends or heavy_backend_warmup_requested()
                ):
                    summary = warmup_configured_heavy_backends(
                        agent, initial_observation
                    )
                    print(f"[WORKER] Heavy backend warmup: {summary}", flush=True)
                all_results.extend(
                    _run_group(
                        episodes,
                        agent=agent,
                        headless=headless,
                        image_dir=image_dir,
                        viewer_mode=viewer_mode,
                        max_steps=FORMAL_EVALUATION_MAX_DECISION_STEPS,
                        output_dir=Path(model_run["output_dir"]),
                        prebuilt_env=env,
                        prebuilt_initial_observation=(
                            initial_observation if model_index == 0 else None
                        ),
                        defer_env_close=True,
                        skip_navmesh=skip_navmesh,
                        room_observer_view_names=room_observer_view_names,
                        room_observer_resolution=room_observer_resolution,
                        room_observer_stride=room_observer_stride,
                        room_observer_max_frames=room_observer_max_frames,
                        agent_harness_config=agent_harness_config,
                        render_mode=render_mode,
                        human_motion_backend=human_motion_backend,
                        evaluation_identity_by_episode={
                            str(row["episode_id"]): {
                                "schema_version": "tarsphere.result_identity.v1",
                                "episode_id": str(row["episode_id"]),
                                "episode_sha256": str(row["sha256"]),
                                "scene_job_id": scene_job_id,
                                "model_name": str(model_run["name"]),
                                "model_id": str(model_run["model"]),
                            }
                            for row in model_run.get("episodes", ())
                        },
                        evaluate_after_rollout=evaluate_after_rollout,
                        record_video=record_video,
                    )
                )
                del agent
                gc.collect()
        finally:
            env.close()
    return all_results


def keep_process_alive_for_reuse() -> None:
    """Keep the process, Isaac env, and model worker alive after validation."""
    print(
        "[WORKER] TARSphere keep-alive enabled; process will stay resident. "
        "Press Ctrl-C or terminate the PID when reuse is finished.",
        flush=True,
    )
    while True:
        time.sleep(3600)


# ── CLI ───────────────────────────────────────────────────────────────────


def _read_episodes_file(path: str) -> list[str]:
    paths: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(line)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Persistent Isaac worker: bake NavMesh once per scene, reuse across episodes.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        metavar="PATH",
        help="Episode YAML (repeatable). At least one is required.",
    )
    parser.add_argument(
        "--episodes-file",
        default=None,
        help="Newline-delimited file of episode YAML paths (one per line, # comments ok).",
    )
    parser.add_argument(
        "--evaluation-schedule",
        type=Path,
        default=None,
        help="Scene-resident multi-model schedule written by the formal launcher.",
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_AGENT_MAX_TOKENS)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--image-dir",
        default="",
        help="Optional directory for per-step RGBD frames. Disabled by default.",
    )
    parser.add_argument(
        "--room-observer-view",
        action="append",
        choices=ROOM_OBSERVER_VIEW_NAMES,
        default=[],
        help=(
            "Policy-invisible third-person view to record; repeat this "
            "argument to record multiple views."
        ),
    )
    parser.add_argument(
        "--room-observer-width",
        type=int,
        default=640,
        help="Width in pixels of each third-person recording frame.",
    )
    parser.add_argument(
        "--room-observer-height",
        type=int,
        default=360,
        help="Height in pixels of each third-person recording frame.",
    )
    parser.add_argument(
        "--room-observer-stride",
        type=int,
        default=1,
        help="Record one third-person frame every N capture opportunities.",
    )
    parser.add_argument(
        "--room-observer-max-frames",
        type=int,
        default=None,
        help="Optional maximum recorded frames per third-person view.",
    )
    parser.add_argument(
        "--viewer",
        choices=sorted(VIEWER_MODES),
        default="third_person",
        help="GUI viewport mode (third_person / first_person / free).",
    )
    parser.add_argument(
        "--human-motion-backend",
        choices=("kinematic", "animation"),
        default="kinematic",
        help="Human execution backend; animation is for demos.",
    )
    parser.add_argument(
        "--render-mode",
        choices=tuple(mode.value for mode in RenderMode),
        default=RenderMode.EVALUATION.value,
        help=(
            "evaluation (default): one ego render per decision, at the "
            "snapshot capture. demo: keep the native cadence between "
            "decisions for smooth video."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Policy decision limit.",
    )
    parser.add_argument(
        "--no-navmesh",
        action="store_true",
        help="Skip AnimPeople NavMesh bake for stationary-Human episodes.",
    )
    parser.add_argument(
        "--warmup-heavy-backends",
        action="store_true",
        help="Force-load configured SAM3 backend after Isaac reset and before the episode loop.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for per-episode result JSON files.",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the Isaac env and model worker process resident after the run.",
    )
    parser.add_argument(
        "--evaluate-after-rollout",
        action="store_true",
        help=(
            "Queue online task scoring after each saved rollout without "
            "blocking Isaac. Failure attribution remains offline."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "Record continuous ego, online-map, subtitle, and decision-panel "
            "videos. Disabled by default for maximum formal throughput."
        ),
    )
    add_agent_harness_arguments(parser)
    args = parser.parse_args()

    episode_paths: list[str] = list(args.episode)
    if args.episodes_file:
        episode_paths.extend(_read_episodes_file(args.episodes_file))
    if not episode_paths and args.evaluation_schedule is None:
        parser.error("Provide episodes or an --evaluation-schedule.")
    if episode_paths and args.evaluation_schedule is not None:
        parser.error("--evaluation-schedule cannot be combined with episode paths.")
    if args.room_observer_width < 1 or args.room_observer_height < 1:
        parser.error("Room-observer width and height must be positive.")
    if len(set(args.room_observer_view)) != len(args.room_observer_view):
        parser.error("Each --room-observer-view may be selected only once.")

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    api_key = (
        args.api_key
        or os.environ.get("TARSPHERE_AGENT_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )

    common = {
        "headless": args.headless,
        "image_dir": args.image_dir,
        "viewer_mode": args.viewer,
        "max_steps": args.max_steps,
        "warmup_heavy_backends": args.warmup_heavy_backends,
        "skip_navmesh": args.no_navmesh,
        "room_observer_view_names": tuple(args.room_observer_view),
        "room_observer_resolution": (
            args.room_observer_width,
            args.room_observer_height,
        ),
        "room_observer_stride": args.room_observer_stride,
        "room_observer_max_frames": args.room_observer_max_frames,
        "agent_harness_config": agent_harness_config_from_args(args),
        "render_mode": args.render_mode,
        "human_motion_backend": args.human_motion_backend,
        "evaluate_after_rollout": args.evaluate_after_rollout,
        "record_video": args.record_video,
    }
    if args.evaluation_schedule is not None:
        if args.keep_alive:
            parser.error("--evaluation-schedule cannot use --keep-alive.")
        results = run_evaluation_schedule(args.evaluation_schedule, **common)
    else:
        results = run_worker(
            episode_paths,
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir,
            keep_alive=(
                args.keep_alive
                or os.environ.get("TARSPHERE_KEEP_WORKER_ALIVE", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            **common,
        )

    print("\n[WORKER] SUMMARY")
    for r in results:
        print(_summary_line(None, r), flush=True)
    n_success = sum(1 for r in results if r.get("success"))
    print(f"  {n_success}/{len(results)} succeeded.", flush=True)
    if args.keep_alive or os.environ.get(
        "TARSPHERE_KEEP_WORKER_ALIVE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        keep_process_alive_for_reuse()


if __name__ == "__main__":
    main()
