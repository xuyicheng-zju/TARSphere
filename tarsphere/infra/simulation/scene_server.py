"""Serve independent episodes in a persistent Isaac Sim scene.

``SceneServer`` retains the stage, optional NavMesh and Replicator pipeline
between requests. Episode resets restore robot and object state and recycle
NPCs without stopping the timeline. Bake-only startup defers NPC spawning until
the first episode. Per-episode output directories separate captured frames
and optional ffmpeg videos."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from tarsphere.harness.factory import (
    build_graspgenx_backend,
    build_robot_agent_envelope,
    build_robot_tool_registry,
)
from tarsphere.harness.harness import (
    AgentHarnessConfig,
)
from tarsphere.harness.warmup import (
    heavy_backend_warmup_requested,
    warmup_configured_heavy_backends,
)
from tarsphere.api_usage import aggregate_model_calls, merge_usage_reports
from tarsphere.configs.secrets import EvaluationSupportServiceError
from tarsphere.evaluation.online_scoring import score_saved_result
from tarsphere.infra.control.navmesh_requirements import assert_navmesh_policy
from tarsphere.infra.control.physical_contracts import (
    G1_GRASP_CERTIFICATION_DISPLACEMENT_M,
    NAVIGATION_MIN_HEADING_PROGRESS_RAD,
    NAVIGATION_MIN_XY_PROGRESS_M,
    SIMULATION_PHYSICS_HZ,
)
from tarsphere.infra.episode.episode_runner import (
    AutonomousEpisodeRunner,
    resolve_agent_action_space,
)
from tarsphere.infra.episode.human_actor_runtime import HumanActorRuntime
from tarsphere.infra.episode.rollout_control import ProgressWatchdog
from tarsphere.infra.perception.continuous_ego_video import (
    ContinuousEgoVideoArtifacts,
)
from tarsphere.infra.perception.room_observer import ROOM_OBSERVER_VIEW_NAMES
from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv, RenderMode
from tarsphere.source_identity import runtime_source_tree_sha256


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "episode"


def _request_artifact_slug(episode_slug: str, request_id: str) -> str:
    """Return the unique filesystem stem for one physical request attempt."""
    return f"{_safe_filename(episode_slug)}_{_safe_filename(request_id)}"


def _write_json_atomic(
    path: Path,
    value: Any,
    *,
    default: Any = None,
) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(
        json.dumps(value, indent=2, default=default) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


def _summary(slug: str, result: dict) -> str:
    return (
        f"[SERVER] {slug}: success={result.get('success')} "
        f"steps={result.get('decision_steps')} "
        f"physics={result.get('physics_steps', '?')} "
        f"renders={result.get('render_count', '?')} "
        f"mode={result.get('render_mode', '?')} "
        f"wall={result.get('wall_clock_s', '?')}s "
        f"video={result.get('video_path', '')}"
    )


class EpisodeBusyError(RuntimeError):
    """Raised when a second rollout is submitted to an occupied scene."""


def _load_object_volumes(episode: Any) -> dict:
    """Load task-relevant GT object volumes from the compiled episode."""
    from tarsphere.geometry import BBox3D

    wk = getattr(episode, "world_knowledge", None) or {}
    objects = wk.get("objects")
    if not isinstance(objects, list) or not objects:
        return {}
    volumes: dict = {}
    for item in objects:
        if not isinstance(item, dict):
            raise TypeError(
                "Every compiled world_knowledge.objects entry must be a mapping."
            )
        oid = str(item.get("object_id") or "").strip()
        if not oid:
            raise RuntimeError("Compiled object is missing object_id.")
        mn = item.get("bounds_min_xyz")
        mx = item.get("bounds_max_xyz")
        if not isinstance(mn, (list, tuple)) or not isinstance(
            mx,
            (list, tuple),
        ):
            raise RuntimeError(
                f"Compiled object {oid!r} is missing world-space bounds."
            )
        try:
            volumes[oid] = BBox3D(
                min=(float(mn[0]), float(mn[1]), float(mn[2])),
                max=(float(mx[0]), float(mx[1]), float(mx[2])),
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(
                f"Compiled object {oid!r} has invalid world-space bounds."
            ) from exc
    return volumes


class SceneServer:
    """A persistent Isaac worker that serves same-scene episodes independently.

    The Isaac env is created lazily on the first ``run_episode`` call (which
    bakes the NavMesh).  Later calls soft-reset without stopping the timeline
    (see ``IsaacAutonomousEnv.reset``), reusing the bake.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        output_dir: str | None = None,
        image_root: Optional[str] = None,
        viewer_mode: str = "first_person",
        agent: Optional[Any] = None,
        api_key: str = "",
        model: str = "",
        api_base: str = "",
        agent_mode: str = "model",
        physical_task_only: bool = False,
        video_fps: int = 4,
        skip_navmesh: bool = False,
        enable_streaming: bool = False,
        enable_human_ego_cameras: bool = False,
        room_observer_views: int = 0,
        room_observer_view_names: tuple[str, ...] = (),
        room_observer_resolution: tuple[int, int] = (640, 360),
        room_observer_stride: int = 1,
        room_observer_max_frames: int = 48,
        continuous_ego_video: bool = False,
        continuous_ego_video_stride: int = 8,
        agent_harness_config: AgentHarnessConfig | None = None,
        render_mode: RenderMode | str = RenderMode.EVALUATION,
        human_motion_backend: str = "kinematic",
        evaluate_after_rollout: bool = False,
    ) -> None:
        self.headless = headless
        self.viewer_mode = viewer_mode
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path.home() / ".local/share/tarsphere/server"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_root = Path(image_root) if image_root else self.output_dir / "images"
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.output_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = int(video_fps)
        if human_motion_backend not in {"kinematic", "animation"}:
            raise ValueError("human_motion_backend must be kinematic or animation")
        self.human_motion_backend = human_motion_backend
        self.evaluate_after_rollout = bool(evaluate_after_rollout)
        self._online_score_executor = (
            ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="scene-online-score",
            )
            if self.evaluate_after_rollout
            else None
        )
        self.skip_navmesh = bool(skip_navmesh or human_motion_backend == "kinematic")
        self.enable_streaming = bool(enable_streaming)
        self.enable_human_ego_cameras = bool(enable_human_ego_cameras)
        requested_observer_names = tuple(room_observer_view_names)
        if requested_observer_names and int(room_observer_views) != 0:
            raise ValueError(
                "Choose explicit room_observer_view_names or a view count, not both."
            )
        if len(set(requested_observer_names)) != len(requested_observer_names):
            raise ValueError("Room observer view names must be unique.")
        unknown_observer_names = sorted(
            set(requested_observer_names) - set(ROOM_OBSERVER_VIEW_NAMES)
        )
        if unknown_observer_names:
            raise ValueError(f"Unknown room observer views: {unknown_observer_names}.")
        self.room_observer_view_names = (
            requested_observer_names
            if requested_observer_names
            else ROOM_OBSERVER_VIEW_NAMES[: int(room_observer_views)]
        )
        self.room_observer_views = len(self.room_observer_view_names)
        self.room_observer_resolution = tuple(
            int(value) for value in room_observer_resolution
        )
        if (
            len(self.room_observer_resolution) != 2
            or min(self.room_observer_resolution) < 1
        ):
            raise ValueError(
                "Room observer resolution must contain two positive integers."
            )
        self.room_observer_stride = int(room_observer_stride)
        self.room_observer_max_frames = int(room_observer_max_frames)
        self.continuous_ego_video = bool(continuous_ego_video)
        self.continuous_ego_video_stride = int(continuous_ego_video_stride)
        if self.room_observer_stride < 1:
            raise ValueError("Room observer stride must be positive.")
        if self.room_observer_max_frames < 1:
            raise ValueError("Room observer maximum frames must be positive.")
        if self.continuous_ego_video_stride < 1:
            raise ValueError("Continuous ego video stride must be positive.")
        self.agent_harness_config = agent_harness_config or AgentHarnessConfig()
        self.render_mode = RenderMode(render_mode)
        # Agent construction is deferred until Isaac is up (see bootstrap() /
        # run_episode). Building it here used to race Isaac for the CUDA
        # context — the SAM3 backend lazily imports torch inside the
        # agent, and if torch initialises CUDA before the Kit runtime the box
        # can freeze on the OOM path. Callers passing a pre-built agent
        # (already initialised against a live Isaac context) bypass the lazy
        # path.
        self.agent = agent
        self._agent_was_supplied = agent is not None
        self._agent_api_key = api_key
        self._agent_model = model
        self._agent_api_base = api_base
        if agent_mode not in {
            "model",
            "golden-oracle",
            "grasp-certification",
            "grasp-asset-certification",
            "physical-task-certification",
        }:
            raise ValueError(f"Unknown scene-server agent mode: {agent_mode!r}")
        self.agent_mode = agent_mode
        self.physical_task_only = bool(physical_task_only)
        if self.physical_task_only and self.agent_mode != "golden-oracle":
            raise ValueError(
                "physical_task_only is available only in golden-oracle mode."
            )
        self.env: Optional[IsaacAutonomousEnv] = None
        self._ep_count = 0
        self._run_lock = threading.Lock()
        self._request_state_lock = threading.Lock()
        self._abort_event = threading.Event()
        self._active_request_id: Optional[str] = None
        self.scene_server_id = str(uuid.uuid4())
        self.scene_server_git_sha = self._git_sha()
        self.scene_server_source_sha256 = self._source_tree_sha256()
        self.scene_server_worktree_dirty = self._worktree_dirty()

    @property
    def active_request_id(self) -> Optional[str]:
        with self._request_state_lock:
            return self._active_request_id

    def abort_active(self, request_id: Optional[str] = None) -> dict[str, Any]:
        """Signal cancellation without touching Isaac from the HTTP thread."""
        with self._request_state_lock:
            active = self._active_request_id
            if active is None:
                return {"aborted": False, "reason": "no_active_episode"}
            if request_id is not None and str(request_id) != active:
                return {
                    "aborted": False,
                    "reason": "request_id_mismatch",
                    "active_request_id": active,
                }
            self._abort_event.set()
            return {"aborted": True, "active_request_id": active}

    def bootstrap(self, episode: Any) -> None:
        """Bring up Isaac and build the agent in the safe order.

        Bake the scene first (this starts the SimulationApp and lets the Kit
        runtime own the CUDA context), THEN build the agent whose SAM3
        backend will lazily import torch. Call this once after construction,
        before the first run_episode(), to pin the load order. After it
        returns, both the env and the agent are live.
        """
        if self.agent is not None:
            # Caller passed a pre-built agent; nothing to defer.
            if self.env is None:
                self.bake_scene(episode)
            return
        if self.env is None:
            self.bake_scene(episode)
        if self.agent is not None:
            return
        print("[SERVER] Building agent after Isaac context is live.", flush=True)
        self.agent = self._build_agent(
            self._agent_api_key,
            self._agent_model,
            self._agent_api_base,
            episode=episode,
        )
        self._warmup_agent_if_requested()

    def _build_agent(
        self,
        api_key: Optional[str],
        model: Optional[str],
        api_base: Optional[str],
        *,
        episode: Any,
    ):
        """Build the envelope-protocol robot agent."""
        episode_actions = resolve_agent_action_space(
            episode,
            self.agent_harness_config,
        )
        if self.agent_mode == "golden-oracle":
            # Golden certification is an authored debug policy. Reload it at
            # episode boundaries so controller fixes can be validated through
            # a soft reset without paying another full Isaac scene bake.
            # Evaluated model agents are deliberately never hot-reloaded.
            importlib.invalidate_caches()
            module = importlib.import_module("tarsphere.harness.certification.oracle")
            source_path = Path(str(module.__file__)).resolve()
            # Isaac's import-hook chain can retain the first SourceFileLoader
            # code object even after invalidate_caches()+reload().  Execute the
            # current source explicitly so a certification log can never claim
            # a controller revision that was not actually run.  This path is
            # benchmark-private and is never used for evaluated model agents.
            source_code = compile(source_path.read_bytes(), str(source_path), "exec")
            exec(source_code, module.__dict__, module.__dict__)
            print(
                "[SERVER] Golden oracle module loaded: "
                f"path={source_path} "
                f"source_mtime_ns={source_path.stat().st_mtime_ns} "
                "rotation_controller="
                f"{getattr(module, '_G1_ROTATION_CONTROLLER_REVISION', 'unknown')}",
                flush=True,
            )
            agent = module.GoldenOracleAgent()
            tools, _ = build_robot_tool_registry(
                agent_harness_config=self.agent_harness_config,
                episode_actions=episode_actions,
            )
            agent.attach_public_tools(tools)
            return agent
        if self.agent_mode == "grasp-certification":
            from tarsphere.harness.certification.grasp_skill_agent import (
                GraspSkillCertificationAgent,
            )

            tools, _ = build_robot_tool_registry(
                agent_harness_config=self.agent_harness_config,
                episode_actions=episode_actions,
            )
            return GraspSkillCertificationAgent(
                agent_harness_config=self.agent_harness_config,
                tools=tools,
                grasp_backend=build_graspgenx_backend(),
            )
        if self.agent_mode == "grasp-asset-certification":
            from tarsphere.harness.certification.grasp_skill_agent import (
                GraspAssetCertificationAgent,
            )

            tools, _ = build_robot_tool_registry(
                agent_harness_config=self.agent_harness_config,
                grasp_grounding="oracle_gt",
                episode_actions=episode_actions,
            )
            return GraspAssetCertificationAgent(
                agent_harness_config=self.agent_harness_config,
                tools=tools,
                grasp_backend=build_graspgenx_backend(),
            )
        if self.agent_mode == "physical-task-certification":
            from tarsphere.harness.certification.grasp_skill_agent import (
                PhysicalTaskCertificationAgent,
            )

            tools, _ = build_robot_tool_registry(
                agent_harness_config=self.agent_harness_config,
                episode_actions=episode_actions,
            )
            return PhysicalTaskCertificationAgent(
                agent_harness_config=self.agent_harness_config,
                tools=tools,
                grasp_backend=build_graspgenx_backend(),
            )
        return build_robot_agent_envelope(
            api_key=api_key or None,
            model=model or None,
            api_base=api_base or None,
            agent_harness_config=self.agent_harness_config,
            episode_actions=episode_actions,
        )

    def _agent_supports_episode(self, episode: Any) -> bool:
        """Whether the current evaluated agent has every admitted tool wired."""

        if self.agent is None:
            return False
        registry = getattr(self.agent, "tools", None)
        spec_names = getattr(registry, "spec_names", None)
        if not callable(spec_names):
            return self._agent_was_supplied
        return set(
            resolve_agent_action_space(
                episode,
                self.agent_harness_config,
            )
        ).issubset(spec_names())

    @property
    def scene_asset_path(self) -> Optional[str]:
        if self.env is not None:
            return self.env.loaded_scene_asset_path
        return None

    def _assert_baked_scene_matches(self, episode: Any) -> None:
        """Reject a request for any scene other than the persistent bake."""

        if self.env is None:
            return
        requested = str(getattr(episode.scene, "asset_path", "") or "")
        loaded = str(self.env.loaded_scene_asset_path or "")
        if requested != loaded:
            raise ValueError(
                "SceneServer scene mismatch: persistent Isaac env was baked "
                f"from {loaded!r}, but the request uses {requested!r}. Start a "
                "new scene server for a different scene."
            )

    def bake_scene(self, episode: Any) -> None:
        """Create the env and bake NavMesh for ``episode.scene`` WITHOUT running
        the episode decision loop.

        Use this on startup when you want the server ready as soon as the scene
        is baked, with the first real episode arriving via a later
        ``run_episode`` call (which soft-resets onto the baked scene).  The
        bootstrap episode is used only for its scene/robot spec; humans and
        agent-ready RGBD warmup are deferred to the first served episode.

        When ``skip_navmesh`` is set, AnimPeople NavMesh bake is skipped (H1
        stationary humans). Episodes that need human ``goto`` fail loud.
        """
        if self.env is not None:
            raise RuntimeError("Scene is already baked.")
        assert_navmesh_policy(
            episode,
            skip_navmesh=self.skip_navmesh,
            human_motion_backend=self.human_motion_backend,
        )
        bake_image_dir = self.image_root / "_bake"
        bake_image_dir.mkdir(parents=True, exist_ok=True)
        nav_msg = (
            "NavMesh: skipped (--no-navmesh)"
            if self.skip_navmesh
            else "NavMesh: enabled (will bake)"
        )
        print(f"[SERVER] Baking scene (no episode run) ... {nav_msg}", flush=True)
        self.env = IsaacAutonomousEnv(
            episode,
            headless=self.headless,
            image_dir=str(bake_image_dir),
            viewer_mode=self.viewer_mode,
            enable_streaming=self.enable_streaming,
            enable_human_ego_cameras=self.enable_human_ego_cameras,
            room_observer_view_names=(self.room_observer_view_names),
            room_observer_resolution=self.room_observer_resolution,
            room_observer_stride=self.room_observer_stride,
            room_observer_max_frames=self.room_observer_max_frames,
            continuous_ego_video=False,
            continuous_ego_video_stride=self.continuous_ego_video_stride,
            agent_harness_config=self.agent_harness_config,
            render_mode=self.render_mode,
            human_motion_backend=self.human_motion_backend,
        )
        # scene_prepare_only: env + optional NavMesh + standing snapshot. Skip
        # human spawn, but prime persistent RTX/RGBD once so later episodes
        # pay only the short soft-reset camera verification.
        self.env.reset(
            scene_prepare_only=True,
            skip_navmesh=self.skip_navmesh,
        )
        # Bake leaves the Kit timeline playing. Without loco commands the G1
        # policy is not stepped and gravity tips the robot before the first
        # /run_episode soft-reset. Hold a standing pose until the episode starts.
        self._hold_standing_robot(reason="post-bake idle hold", settle_steps=4)
        print(
            f"[SERVER] Scene baked: {getattr(episode.scene, 'asset_path', None)}",
            flush=True,
        )
        if self.agent is None and heavy_backend_warmup_requested():
            print("[SERVER] Building agent after Isaac context is live.", flush=True)
            self.agent = self._build_agent(
                self._agent_api_key,
                self._agent_model,
                self._agent_api_base,
                episode=episode,
            )
            self._warmup_agent_if_requested()
            self._hold_standing_robot(
                reason="post-agent-warmup idle hold", settle_steps=4
            )

    def run_episode(
        self,
        episode: Any,
        *,
        max_steps: Optional[int] = None,
        record_video: bool = True,
        request_id: Optional[str] = None,
        certification_stage: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Run exactly one episode; reject concurrent callers instead of queuing."""
        resolved_request_id = str(request_id or uuid.uuid4())
        with self._exclusive_request(resolved_request_id):
            if any(value is not None for value in (api_key, model, api_base)):
                self.configure_model_agent(
                    api_key=api_key,
                    model=model,
                    api_base=api_base,
                )
            return self._run_episode(
                episode,
                max_steps=max_steps,
                record_video=record_video,
                request_id=resolved_request_id,
                certification_stage=certification_stage,
            )

    def configure_model_agent(
        self,
        *,
        api_key: Optional[str],
        model: Optional[str],
        api_base: Optional[str],
    ) -> None:
        """Select the evaluated remote model without reloading the scene."""

        if self.agent_mode != "model":
            raise ValueError("Per-request model selection requires model mode.")
        if self._agent_was_supplied:
            raise ValueError("A caller-supplied agent cannot be replaced.")
        resolved_key = str(api_key or self._agent_api_key or "").strip()
        resolved_model = str(model or self._agent_model or "").strip()
        resolved_base = str(api_base or self._agent_api_base or "").strip()
        if not resolved_key or not resolved_model or not resolved_base:
            raise ValueError(
                "Model selection requires an API key, model ID, and API base."
            )
        if (
            resolved_key == self._agent_api_key
            and resolved_model == self._agent_model
            and resolved_base == self._agent_api_base
        ):
            return
        self._agent_api_key = resolved_key
        self._agent_model = resolved_model
        self._agent_api_base = resolved_base
        self.agent = None
        print(
            "[SERVER] Agent backend selected for next episode: "
            f"model={resolved_model} api_base={resolved_base}",
            flush=True,
        )

    def verify_human_action(
        self,
        *,
        human_id: Optional[str] = None,
        action: dict[str, Any],
        hold_sim_seconds: float = 12.0,
        record_video: bool = True,
        capture_stride: Optional[int] = None,
        physics_ticks_per_sample: int = 8,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Play one Human action on the live baked scene and record it.

        Exists so Human-animation verification stops paying the ~10 min scene
        load per action. Interactive Human-action diagnostics do the
        same work from a cold start; this is that runner's body hoisted onto
        the persistent server so a whole action list costs one bake.

        Video comes from ``RoomObserverRecorder``, which only samples inside a
        native Kit render tick, so this needs the server started with
        ``--render-mode demo``: evaluation mode suppresses mid-window renders
        and would yield an empty frame dir.  The observer is pointed at the
        Human (``room_observer_focus_human``) because the robot never moves
        here and its framing leaves the character off-screen.

        Returns per-sample state plus the peak instantaneous speed.  Peak is
        reported alongside the mean deliberately: a mean silently conflates
        "the animation clock is slow" with "the path follower stalled near its
        target", which are different faults and were confused once already.
        """
        resolved_request_id = str(request_id or uuid.uuid4())
        with self._exclusive_request(resolved_request_id):
            env = self.env
            if env is None:
                raise RuntimeError(
                    "Scene server has no live environment; bake a scene first."
                )
            if not env._human_positions:
                raise RuntimeError(
                    "No Humans are spawned on the live scene. Start the server "
                    "without --bake-only, or run an episode first, so the "
                    "characters exist."
                )
            resolved_human = str(human_id or next(iter(env._human_positions)))
            if resolved_human not in env._human_positions:
                raise RuntimeError(
                    f"Unknown human_id {resolved_human!r}; live characters are "
                    f"{sorted(env._human_positions)}."
                )

            command = str(action.get("command") or "").strip()
            if not command:
                raise ValueError('action requires a "command" field.')
            if record_video and not self.room_observer_view_names:
                raise RuntimeError(
                    "verify_human_action needs room-observer views to record: "
                    "restart the server with --room-observer-views N (and "
                    "--render-mode demo, since the recorder only samples "
                    "inside a native Kit render tick)."
                )
            if capture_stride is not None and int(capture_stride) != int(
                self.room_observer_stride
            ):
                raise ValueError(
                    f"capture_stride {capture_stride} does not match this "
                    f"server's --room-observer-stride "
                    f"{self.room_observer_stride}. The observer stride is "
                    "fixed when the recorder is built, so it cannot be set "
                    "per request; restart the server to change it."
                )

            slug = _safe_filename(f"human_{command}_{resolved_request_id[:8]}")
            image_dir = self._prepare_episode_outputs(slug)

            previous_focus = getattr(env, "room_observer_focus_human", None)
            previous_image_dir = getattr(env, "image_dir", None)
            env.room_observer_focus_human = resolved_human
            if record_video:
                env.image_dir = image_dir
                env._start_room_observer_episode()

            try:
                result = env.apply_human_action(resolved_human, dict(action))
                samples = self._play_human_action(
                    env,
                    resolved_human,
                    hold_sim_seconds=float(hold_sim_seconds),
                    physics_ticks_per_sample=int(physics_ticks_per_sample),
                )
            finally:
                env.room_observer_focus_human = previous_focus
                if record_video:
                    env.image_dir = previous_image_dir

            speeds = [s["speed_m_s"] for s in samples if s.get("speed_m_s") is not None]
            travelled = 0.0
            for previous, current in zip(samples, samples[1:]):
                a, b = previous.get("position"), current.get("position")
                if a and b and len(a) >= 2 and len(b) >= 2:
                    travelled += math.dist(a[:2], b[:2])

            report: dict[str, Any] = {
                "request_id": resolved_request_id,
                "human_id": resolved_human,
                "action": dict(action),
                "applied": bool(getattr(result, "executed", False)),
                "feedback": getattr(result, "feedback", ""),
                "hold_sim_seconds": float(hold_sim_seconds),
                "samples": samples,
                "path_length_m": round(travelled, 4),
                "peak_speed_m_s": round(max(speeds), 3) if speeds else None,
                "mean_speed_m_s": (
                    round(travelled / float(hold_sim_seconds), 3)
                    if hold_sim_seconds > 0
                    else None
                ),
                "final_animation_state": (
                    samples[-1].get("animation_state") if samples else None
                ),
                "final_action_status": (
                    samples[-1].get("action_status") if samples else None
                ),
                "kit_updates": sum(int(s.get("kit_updates") or 0) for s in samples),
            }
            if record_video:
                report["videos"] = self._render_room_observer_videos(slug, image_dir)
            result_path = self.output_dir / f"{slug}.json"
            result_path.write_text(json.dumps(report, indent=2))
            report["result_path"] = str(result_path)
            print(
                f"[SERVER] {slug}: applied={report['applied']} "
                f"anim={report['final_animation_state']} "
                f"status={report['final_action_status']} "
                f"peak={report['peak_speed_m_s']} m/s "
                f"mean={report['mean_speed_m_s']} m/s "
                f"path={report['path_length_m']} m "
                f"pumps={report['kit_updates']} "
                f"videos={list((report.get('videos') or {}).values())}",
                flush=True,
            )
            return report

    def _play_human_action(
        self,
        env: Any,
        human_id: str,
        *,
        hold_sim_seconds: float,
        physics_ticks_per_sample: int,
    ) -> list[dict[str, Any]]:
        """Advance simulated time while one Human action plays.

        The budget is *simulated* seconds, not wall-clock: rendering observer
        views runs the scene far slower than real time, so a wall-clock budget
        truncates the animation mid-action.

        The robot holds zero velocity: this verifies Human animation, and
        ``step_simulation`` is the only stepping path that both ticks
        ``omni.anim.people`` and drives the observer's per-tick capture hooks.
        """
        zero_speed = {"move_by_speed": [0.0, 0.0, 0.0]}
        samples: list[dict[str, Any]] = []
        target_ticks = int(round(float(hold_sim_seconds) * SIMULATION_PHYSICS_HZ))
        ticks_done = 0
        previous_position: Optional[list[float]] = None
        while ticks_done < target_ticks:
            result = env._robot_body.step_simulation(
                env=env._env,
                action_name="wait",
                env_action=zero_speed,
                physics_step_budget=int(physics_ticks_per_sample),
                step_human_bodies=env._step_human_bodies,
                pump_kit_updates=env._pump_kit_updates,
                pump_motion_window=env._pump_human_motion_window,
                global_tick=env._physics_steps,
                diagnose_physics_tick=env._on_physics_tick,
            )
            if result.observation:
                env._obs = result.observation
            stepped = int(result.physics_steps)
            env._physics_steps += stepped
            ticks_done += stepped

            state = (env.get_world_snapshot([human_id]).get("humans") or {}).get(
                human_id, {}
            )
            position = state.get("position")
            speed = None
            if (
                previous_position is not None
                and position is not None
                and len(position) >= 2
                and stepped > 0
            ):
                speed = round(
                    math.dist(position[:2], previous_position[:2])
                    / (stepped / SIMULATION_PHYSICS_HZ),
                    3,
                )
            previous_position = list(position) if position else previous_position
            samples.append(
                {
                    "sim_seconds": round(ticks_done / SIMULATION_PHYSICS_HZ, 3),
                    "position": position,
                    "yaw_rad": state.get("yaw_rad"),
                    "animation_state": state.get("animation_state"),
                    "active_action": state.get("active_action"),
                    "action_status": state.get("action_status"),
                    "kit_updates": int(result.kit_updates),
                    "speed_m_s": speed,
                }
            )
        return samples

    def probe_placement_surfaces(
        self,
        *,
        scene_id: str,
        target_object_id: str,
        destination_object_id: str,
        search: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Probe live destination collision patches without running an episode."""
        from tarsphere.bench.certification.placement import (
            PLACEMENT_SURFACE_SEARCH,
            PlacementSurfaceSearch,
            find_placement_surface_patches,
        )
        from tarsphere.bench.grounding import resolve_scene
        from tarsphere.bench.spatial import (
            discover_manipulation_targets,
            discover_placement_destinations,
        )
        from tarsphere.infra.simulation.scene_query import (
            PhysXSurfaceRaycast,
        )

        resolved_request_id = str(request_id or uuid.uuid4())
        with self._exclusive_request(resolved_request_id):
            if self.env is None:
                raise RuntimeError("Placement surface probing requires a baked scene")
            loaded_asset = str(self.scene_asset_path or "")
            if scene_id not in loaded_asset:
                raise ValueError(
                    f"Placement probe scene {scene_id!r} is not loaded: "
                    f"{loaded_asset!r}"
                )
            scene = resolve_scene(scene_id)
            objects = {item.object_id: item for item in scene.objects}
            try:
                target_object = objects[target_object_id]
                destination_object = objects[destination_object_id]
            except KeyError as exc:
                raise ValueError(
                    f"Placement probe references an unknown grounded object: "
                    f"{exc.args[0]!r}"
                ) from exc

            target_candidates = {
                item.target_object_id: item
                for item in discover_manipulation_targets(scene).targets
            }
            target_candidate = target_candidates.get(target_object_id)
            if target_candidate is None:
                raise ValueError(
                    f"Object is not an eligible manipulation target: "
                    f"{target_object_id!r}"
                )
            destination_ids = {
                item.destination_object_id
                for item in discover_placement_destinations(
                    scene,
                    target_candidate,
                ).destinations
            }
            if destination_object_id not in destination_ids:
                raise ValueError(
                    f"Object is not an eligible destination for the target: "
                    f"{destination_object_id!r}"
                )
            live_bounds = getattr(
                self.env,
                "live_scene_object_bounds",
                None,
            )
            if not callable(live_bounds):
                raise RuntimeError(
                    "Placement surface probing requires live_scene_object_bounds"
                )

            search_contract = (
                PLACEMENT_SURFACE_SEARCH
                if search is None
                else PlacementSurfaceSearch.model_validate(search)
            )
            patches = find_placement_surface_patches(
                target_object,
                destination_object,
                target_bounds_xyz=live_bounds(target_object.runtime_prim),
                raycast=PhysXSurfaceRaycast(),
                search=search_contract,
            )
            return {
                "request_id": resolved_request_id,
                "scene_id": scene_id,
                "target_object_id": target_object_id,
                "destination_object_id": destination_object_id,
                "search": search_contract.model_dump(mode="json"),
                "patches": [patch.model_dump(mode="json") for patch in patches],
                "scene_server_id": self.scene_server_id,
                "scene_server_git_sha": self.scene_server_git_sha,
                "scene_server_source_sha256": self.scene_server_source_sha256,
                "scene_server_worktree_dirty": self.scene_server_worktree_dirty,
            }

    def probe_source_support(
        self,
        *,
        scene_id: str,
        target_object_id: str,
        search: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Identify the target's live source support without running an episode."""
        from tarsphere.bench.certification.source_support import (
            SOURCE_SUPPORT_SEARCH,
            SourceSupportCertificationError,
            SourceSupportSearch,
            certify_source_support,
        )
        from tarsphere.bench.grounding import resolve_scene
        from tarsphere.bench.spatial import (
            discover_manipulation_targets,
        )
        from tarsphere.infra.simulation.scene_query import (
            PhysXSurfaceRaycast,
        )

        resolved_request_id = str(request_id or uuid.uuid4())
        with self._exclusive_request(resolved_request_id):
            if self.env is None:
                raise RuntimeError("Source-support probing requires a baked scene")
            loaded_asset = str(self.scene_asset_path or "")
            if scene_id not in loaded_asset:
                raise ValueError(
                    f"Source-support probe scene {scene_id!r} is not loaded: "
                    f"{loaded_asset!r}"
                )
            scene = resolve_scene(scene_id)
            objects = {item.object_id: item for item in scene.objects}
            targets = {
                item.target_object_id: item
                for item in discover_manipulation_targets(scene).targets
            }
            target = targets.get(target_object_id)
            if target is None:
                raise ValueError(
                    f"Object is not an eligible manipulation target: "
                    f"{target_object_id!r}"
                )
            target_object = objects[target_object_id]
            if not target_object.runtime_prim:
                raise ValueError(
                    "Eligible manipulation target lacks a runtime prim: "
                    f"{target_object_id!r}"
                )
            live_bounds = getattr(
                self.env,
                "live_scene_object_bounds",
                None,
            )
            if not callable(live_bounds):
                raise RuntimeError(
                    "Source-support probing requires live_scene_object_bounds"
                )
            search_contract = (
                SOURCE_SUPPORT_SEARCH
                if search is None
                else SourceSupportSearch.model_validate(search)
            )
            try:
                evidence = certify_source_support(
                    scene,
                    target,
                    target_bounds_xyz=live_bounds(target_object.runtime_prim),
                    raycast_all=PhysXSurfaceRaycast().all_hits,
                    search=search_contract,
                )
            except SourceSupportCertificationError as exc:
                return {
                    "request_id": resolved_request_id,
                    "scene_id": scene_id,
                    "target_object_id": target_object_id,
                    "search": search_contract.model_dump(mode="json"),
                    "rejection": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "scene_server_id": self.scene_server_id,
                    "scene_server_git_sha": self.scene_server_git_sha,
                    "scene_server_source_sha256": self.scene_server_source_sha256,
                    "scene_server_worktree_dirty": self.scene_server_worktree_dirty,
                }
            return {
                "request_id": resolved_request_id,
                "scene_id": scene_id,
                "search": search_contract.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "scene_server_id": self.scene_server_id,
                "scene_server_git_sha": self.scene_server_git_sha,
                "scene_server_source_sha256": self.scene_server_source_sha256,
                "scene_server_worktree_dirty": self.scene_server_worktree_dirty,
            }

    @contextmanager
    def _exclusive_request(self, request_id: str):
        """Own the single Isaac request slot for one bounded operation."""
        if not self._run_lock.acquire(blocking=False):
            raise EpisodeBusyError(
                "Scene server already has an active operation: "
                f"{self.active_request_id!r}."
            )
        with self._request_state_lock:
            self._active_request_id = request_id
            self._abort_event.clear()
        try:
            yield
        finally:
            with self._request_state_lock:
                self._active_request_id = None
                self._abort_event.clear()
            self._run_lock.release()

    def _run_episode(
        self,
        episode: Any,
        *,
        max_steps: Optional[int],
        record_video: bool,
        request_id: str,
        certification_stage: Optional[str],
    ) -> dict:
        """Run one episode on the persistent scene; return its result dict.

        The first call bakes; subsequent calls soft-reset and reuse the bake.
        Robot start + human count/positions come from ``episode``; each episode
        is independent of the others.
        """
        self._assert_baked_scene_matches(episode)
        slug = _safe_filename(
            getattr(episode, "episode_id", f"episode_{self._ep_count}")
        )
        attempt_slug = _request_artifact_slug(slug, request_id)
        # Re-running the same episode id must not mix fresh frames with stale
        # frames from an earlier run. ffmpeg consumes a glob, so stale PNGs can
        # otherwise make a video look like the camera suddenly teleported.
        ep_image_dir = self._prepare_episode_outputs(attempt_slug)
        # Private online-map evidence is isolated per episode just like RGBD
        # frames.  It is never added to the Agent prompt; the artifacts exist
        # so occupancy growth and collision margins can be audited offline.
        online_map_dir = self.output_dir / "online_maps" / attempt_slug
        if online_map_dir.exists():
            if not online_map_dir.is_dir():
                raise RuntimeError(
                    f"Episode online-map path is not a directory: {online_map_dir}"
                )
            shutil.rmtree(online_map_dir)
        online_map_dir.mkdir(parents=True, exist_ok=False)
        os.environ["TARSPHERE_RECORD_DIR"] = str(online_map_dir)
        assert_navmesh_policy(
            episode,
            skip_navmesh=self.skip_navmesh,
            human_motion_backend=self.human_motion_backend,
        )

        if self.env is None:
            nav_msg = (
                "skipping NavMesh (--no-navmesh)"
                if self.skip_navmesh
                else "baking NavMesh"
            )
            print(
                f"[SERVER] First episode on scene — creating env + {nav_msg}.",
                flush=True,
            )
            self.env = IsaacAutonomousEnv(
                episode,
                headless=self.headless,
                image_dir=str(ep_image_dir),
                viewer_mode=self.viewer_mode,
                enable_streaming=self.enable_streaming,
                enable_human_ego_cameras=self.enable_human_ego_cameras,
                room_observer_view_names=self.room_observer_view_names,
                room_observer_resolution=self.room_observer_resolution,
                room_observer_stride=self.room_observer_stride,
                room_observer_max_frames=self.room_observer_max_frames,
                continuous_ego_video=False,
                continuous_ego_video_stride=self.continuous_ego_video_stride,
                agent_harness_config=self.agent_harness_config,
                render_mode=self.render_mode,
                human_motion_backend=self.human_motion_backend,
            )
            # Bring Isaac fully up before building the agent. Then arm a soft
            # reset so the runner starts the episode on the already-baked scene.
            self.env.reset(skip_navmesh=self.skip_navmesh)
            if self.agent is None:
                print(
                    "[SERVER] Building agent after Isaac context is live.",
                    flush=True,
                )
                self.agent = self._build_agent(
                    self._agent_api_key,
                    self._agent_model,
                    self._agent_api_base,
                    episode=episode,
                )
                self._warmup_agent_if_requested()
            self._hold_standing_robot(
                reason="pre-first-episode idle hold", settle_steps=4
            )
            self.env.image_dir = ep_image_dir
            self.env.load_episode(episode)
        else:
            reuse_msg = (
                "no NavMesh on this server (--no-navmesh)"
                if self.skip_navmesh
                else "reusing baked NavMesh"
            )
            print(
                f"[SERVER] Same scene — soft reset (GoTo-dedupe human recycle, "
                f"{reuse_msg}).",
                flush=True,
            )
            self.env.image_dir = ep_image_dir
            # Re-assert standing before arming soft reset: bake→HTTP idle can tip
            # G1 while the timeline is live without loco commands.
            self._hold_standing_robot(reason="pre-soft-reset idle hold", settle_steps=4)
            self.env.load_episode(episode)

        if self.agent_mode == "golden-oracle":
            # Reload exactly once at every episode boundary.  The Golden
            # module is intentionally hot-reloadable; evaluated agents keep
            # their long-lived backend instance.
            self.agent = self._build_agent(
                self._agent_api_key,
                self._agent_model,
                self._agent_api_base,
                episode=episode,
            )
        elif not self._agent_supports_episode(episode):
            print(
                "[SERVER] Building episode-scoped agent tool surface after "
                "Isaac context is live.",
                flush=True,
            )
            self.agent = self._build_agent(
                self._agent_api_key,
                self._agent_model,
                self._agent_api_base,
                episode=episode,
            )
            self._warmup_agent_if_requested()

        resolved_certification_stage = (
            str(certification_stage or "C").strip().upper()
            if self.agent_mode == "golden-oracle"
            else None
        )
        if certification_stage is not None and self.agent_mode != "golden-oracle":
            raise ValueError(
                "certification_stage is available only in golden-oracle mode."
            )
        configure_certification_stage = getattr(
            self.agent, "configure_certification_stage", None
        )
        if resolved_certification_stage is not None:
            if not callable(configure_certification_stage):
                raise TypeError(
                    "Golden Oracle must implement configure_certification_stage()."
                )
            configure_certification_stage(resolved_certification_stage)
        if self.physical_task_only:
            configure_physical_task_only = getattr(
                self.agent, "configure_physical_task_only", None
            )
            if not callable(configure_physical_task_only):
                raise TypeError(
                    "Golden Oracle must implement configure_physical_task_only()."
                )
            configure_physical_task_only(True)

        self.env.configure_continuous_ego_video(
            enabled=(self.continuous_ego_video and record_video),
            output_dir=self.video_dir,
            slug=attempt_slug,
            capture_stride=self.continuous_ego_video_stride,
            include_bev=False,
        )

        heartbeat_path = self.output_dir / f"{attempt_slug}.heartbeat.jsonl"
        model_output_path = self.output_dir / f"{attempt_slug}.model_output.jsonl"
        watchdog = ProgressWatchdog(
            min_xy_progress_m=float(
                os.environ.get(
                    "TARSPHERE_ROLLOUT_WATCHDOG_MIN_XY_M",
                    str(NAVIGATION_MIN_XY_PROGRESS_M),
                )
            ),
            min_heading_progress_rad=float(
                os.environ.get(
                    "TARSPHERE_ROLLOUT_WATCHDOG_MIN_HEADING_RAD",
                    str(NAVIGATION_MIN_HEADING_PROGRESS_RAD),
                )
            ),
        )
        runner = AutonomousEpisodeRunner(
            episode=episode,
            environment=self.env,
            agent=self.agent,
            human=HumanActorRuntime(
                episode,
            ),
            collector=self._build_collector(episode, self.env),
            abort_requested=self._abort_event.is_set,
            heartbeat_path=heartbeat_path,
            model_output_path=model_output_path,
            progress_watchdog=watchdog,
            agent_harness_config=self.agent_harness_config,
        )
        t0 = time.monotonic()
        rollout_infrastructure_error: BaseException | None = None
        try:
            result = runner.run(max_decision_steps=max_steps)
        except EvaluationSupportServiceError as exc:
            rollout_infrastructure_error = exc
            print(
                "[SERVER] Evaluation-support service failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            result = runner.partial_result(failure=exc)
            result["outcome_status"] = "infrastructure_invalid"
            result["infrastructure_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            print(
                f"[SERVER] Rollout failed loud: {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            result = runner.partial_result(failure=exc)
            rollout_infrastructure_error = exc
            result["outcome_status"] = "infrastructure_invalid"
            result["infrastructure_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        if result.get("termination_reason") in {
            "model_service_retry_exhausted",
            "model_request_failed",
        }:
            transport_exhausted = result.get("termination_reason") == (
                "model_service_retry_exhausted"
            )
            rollout_infrastructure_error = RuntimeError(
                "The evaluated Agent endpoint exhausted transport recovery."
                if transport_exhausted
                else "The evaluated Agent request failed before a provider "
                "response; check its model binding and request contract."
            )
            result["success"] = False
            result["outcome_status"] = "infrastructure_invalid"
            result["infrastructure_error"] = {
                "type": (
                    "AgentModelServiceError"
                    if transport_exhausted
                    else "AgentModelRequestError"
                ),
                "message": str(rollout_infrastructure_error),
            }
        result["wall_clock_s"] = round(time.monotonic() - t0, 1)
        result["episode_id"] = getattr(episode, "episode_id", slug)
        result["scene_id"] = str(
            (getattr(episode, "world_knowledge", None) or {}).get("scene_id") or ""
        )
        result["agent_mode"] = self.agent_mode
        result["agent_model"] = self._agent_model
        result["human_motion_backend"] = self.human_motion_backend
        result["image_dir"] = str(ep_image_dir)
        result["request_id"] = request_id
        result["physical_attempt_slug"] = attempt_slug
        result["scene_server_id"] = self.scene_server_id
        result["scene_server_git_sha"] = self.scene_server_git_sha
        result["scene_server_source_sha256"] = self.scene_server_source_sha256
        result["scene_server_worktree_dirty"] = self.scene_server_worktree_dirty
        result["loaded_scene_asset_path"] = self.env.loaded_scene_asset_path
        result["loaded_scene_usd_path"] = str(self.env.loaded_scene_usd_path or "")
        result["model_output_trace_path"] = str(model_output_path)
        continuous_ego_artifacts = None
        try:
            continuous_ego_artifacts = self.env.finish_continuous_ego_video()
        except Exception as exc:  # noqa: BLE001
            result["continuous_ego_video_error"] = str(exc)
            print(
                f"[SERVER] continuous ego video finalization failed: {exc}",
                flush=True,
            )
        if heartbeat_path.is_file():
            result["heartbeat_path"] = str(heartbeat_path)
            result["heartbeat_sha256"] = hashlib.sha256(
                heartbeat_path.read_bytes()
            ).hexdigest()
        if model_output_path.is_file():
            result["model_output_trace_sha256"] = hashlib.sha256(
                model_output_path.read_bytes()
            ).hexdigest()
        if continuous_ego_artifacts is not None:
            result["video"] = {
                "path": str(continuous_ego_artifacts.ego_video_path),
                "sha256": hashlib.sha256(
                    continuous_ego_artifacts.ego_video_path.read_bytes()
                ).hexdigest(),
                "view": "ego",
                "manifest_path": str(continuous_ego_artifacts.manifest_path),
                "manifest_sha256": hashlib.sha256(
                    continuous_ego_artifacts.manifest_path.read_bytes()
                ).hexdigest(),
                "resolution": list(continuous_ego_artifacts.resolution),
                "fps": continuous_ego_artifacts.fps,
                "frame_count": len(continuous_ego_artifacts.physics_steps),
            }
        if resolved_certification_stage is not None:
            if (
                result.get("termination_reason") == "decision_budget_exhausted"
                and "failure" not in result
            ):
                result["failure"] = {
                    "type": "DecisionBudgetExceeded",
                    "message": (
                        "Golden Oracle reached the rollout safety ceiling "
                        "before completing its certification gate."
                    ),
                    "details": {"max_steps": max_steps},
                }
            result["certification_stage"] = resolved_certification_stage
            result.update(
                self._certification_result(
                    episode=episode,
                    result=result,
                    runner=runner,
                    stage=resolved_certification_stage,
                )
            )

        # Physical quality does not depend on semantic judge availability.
        # No reference is silently inferred from a pending Oracle artifact.
        if result.get("trace"):
            from tarsphere.bench.execution_metrics import attach_execution_quality

            attach_execution_quality(result, episode=episode, environment=self.env)

        # Required presentation artifacts are part of the formal physical
        # attempt. Finish them before freezing the rollout or spending Judge
        # tokens. Optional room-observer diagnostics never determine validity.
        required_video_error = ""
        if record_video:
            try:
                if continuous_ego_artifacts is not None:
                    continuous_videos = self._render_continuous_ego_videos(
                        slug=attempt_slug,
                        artifacts=continuous_ego_artifacts,
                        decision_trace=result.get("trace") or (),
                        heartbeats=runner.heartbeats,
                    )
                    result["continuous_ego_videos"] = continuous_videos
                    result["continuous_ego_video_sha256"] = {
                        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        for name, path in continuous_videos.items()
                    }
                    result["video_path"] = continuous_videos["ego_decision_panel"]
                elif episode.evaluation is None:
                    result["video_path"] = str(
                        self._render_video(attempt_slug, ep_image_dir)
                    )
                else:
                    required_video_error = "continuous ego recording was unavailable"
            except Exception as exc:  # noqa: BLE001
                required_video_error = str(exc)
                print(f"[SERVER] ego video rendering failed: {exc}", flush=True)
            try:
                observer_videos = self._render_room_observer_videos(
                    attempt_slug,
                    ep_image_dir,
                )
                if observer_videos:
                    result["room_observer_videos"] = observer_videos
            except Exception as exc:  # noqa: BLE001
                result["observer_video_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                print(
                    f"[SERVER] room-observer video rendering failed: {exc}",
                    flush=True,
                )
        else:
            required_video_error = "record_video was disabled"

        missing_artifacts: list[str] = []
        if episode.evaluation is not None:
            missing_artifacts = [
                name
                for name, present in (
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
                    (
                        "raw_ego_video",
                        bool(
                            isinstance(result.get("video"), dict)
                            and result["video"].get("path")
                            and result["video"].get("sha256")
                            and result["video"].get("manifest_path")
                            and result["video"].get("manifest_sha256")
                        ),
                    ),
                    (
                        "ego_decision_panel",
                        bool(
                            isinstance(result.get("continuous_ego_videos"), dict)
                            and result["continuous_ego_videos"].get(
                                "ego_decision_panel"
                            )
                        ),
                    ),
                    (
                        "online_map_video",
                        bool(
                            isinstance(result.get("continuous_ego_videos"), dict)
                            and result["continuous_ego_videos"].get("online_map")
                        ),
                    ),
                    (
                        "model_output_ass",
                        bool(
                            isinstance(result.get("continuous_ego_videos"), dict)
                            and result["continuous_ego_videos"].get("model_output_ass")
                        ),
                    ),
                )
                if not present
            ]
        if episode.evaluation is not None and (
            required_video_error or missing_artifacts
        ):
            artifact_error = required_video_error or (
                "missing " + ", ".join(missing_artifacts)
            )
            if required_video_error:
                result["video_error"] = required_video_error
            result["success"] = False
            result["outcome_status"] = "infrastructure_invalid"
            result["infrastructure_error"] = {
                "type": "RequiredFormalArtifactError",
                "message": (
                    "Formal episode is missing required forensic artifacts: "
                    f"{artifact_error}"
                ),
            }
            if rollout_infrastructure_error is None:
                rollout_infrastructure_error = RuntimeError(
                    result["infrastructure_error"]["message"]
                )

        # Preserve every request-scoped physical attempt before any Judge call.
        # The stable ``<episode>.json`` below remains a latest-result view; this
        # immutable directory is the forensic source for failure attribution.
        attempt_dir = self.output_dir / "artifacts" / slug / _safe_filename(request_id)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        episode_path = attempt_dir / "episode.json"
        _write_json_atomic(
            episode_path,
            episode.model_dump(mode="json"),
            default=str,
        )
        result["episode_artifact_path"] = str(episode_path)
        result["episode_artifact_sha256"] = hashlib.sha256(
            episode_path.read_bytes()
        ).hexdigest()
        rollout_path = attempt_dir / "rollout.json"
        _write_json_atomic(
            rollout_path,
            {
                "schema_version": "tarsphere.rollout_artifact.v1",
                "episode_id": result.get("episode_id"),
                "request_id": request_id,
                "episode_artifact_path": result["episode_artifact_path"],
                "episode_artifact_sha256": result["episode_artifact_sha256"],
                "termination_reason": result.get("termination_reason"),
                "decision_steps": result.get("decision_steps"),
                "action_steps": result.get("action_steps"),
                "physics_steps": result.get("physics_steps"),
                "trace": list(result.get("trace") or ()),
                "human_replay_log": list(result.get("human_replay_log") or ()),
                "heartbeats": list(getattr(runner, "heartbeats", ())),
                "evaluation_steps": [
                    step.model_dump(mode="json")
                    for step in getattr(runner, "_evaluation_steps", ())
                ],
                "heartbeat_path": result.get("heartbeat_path"),
                "heartbeat_sha256": result.get("heartbeat_sha256"),
                "model_output_trace_path": result.get("model_output_trace_path"),
                "model_output_trace_sha256": result.get("model_output_trace_sha256"),
                "video": result.get("video"),
                "continuous_ego_videos": result.get("continuous_ego_videos"),
                "continuous_ego_video_sha256": result.get(
                    "continuous_ego_video_sha256"
                ),
                "failure": result.get("failure"),
            },
            default=str,
        )
        result["rollout_artifact_path"] = str(rollout_path)
        result["rollout_artifact_sha256"] = hashlib.sha256(
            rollout_path.read_bytes()
        ).hexdigest()

        # Official TARSphere analysis reads the same uninterrupted trajectory.
        # Evaluator failure invalidates the report, not the physical run file.
        if (
            resolved_certification_stage in {None, "C"}
            and episode.evaluation is not None
            and rollout_infrastructure_error is None
        ):
            try:
                trajectory_path = attempt_dir / "evaluation_trajectory.json"
                _write_json_atomic(
                    trajectory_path,
                    runner.evaluation_trajectory().model_dump(mode="json"),
                )
                result["evaluation_trajectory_path"] = str(trajectory_path)
                result["evaluation_trajectory_sha256"] = hashlib.sha256(
                    trajectory_path.read_bytes()
                ).hexdigest()
                result["evaluation_attempts"] = 0
                result["success"] = False
                result["outcome_status"] = "pending_evaluation"
                _write_json_atomic(
                    self.output_dir / f"{slug}.json", result, default=str
                )
                _write_json_atomic(attempt_dir / "result.json", result, default=str)
            except Exception as exc:  # noqa: BLE001
                print(f"[SERVER] TARSphere evaluation failed: {exc}", flush=True)
                result["evaluation_report"] = {"error": str(exc)}
                result["success"] = False
                result["outcome_status"] = "evaluation_failed"

        agent_usage = result.get("agent_model_usage")
        if isinstance(agent_usage, dict):
            evaluation_usage = result.get("evaluation_model_usage")
            if not isinstance(evaluation_usage, dict):
                evaluation_usage = aggregate_model_calls([])
                result["evaluation_model_usage"] = evaluation_usage
            human_usage = result.get("human_model_usage")
            if not isinstance(human_usage, dict):
                human_usage = aggregate_model_calls([])
                result["human_model_usage"] = human_usage
            result["api_usage"] = merge_usage_reports(
                (
                    agent_usage,
                    human_usage,
                    evaluation_usage,
                )
            )

        if self.agent_mode in {
            "grasp-asset-certification",
            "physical-task-certification",
        }:
            status_fn = getattr(self.agent, "certification_status", None)
            if not callable(status_fn):
                raise TypeError(
                    "Physical certification must expose certification_status()."
                )
            result[
                "grasp_asset_certification"
                if self.agent_mode == "grasp-asset-certification"
                else "physical_task_certification"
            ] = dict(status_fn())

        # Write the result JSON AFTER the video path is known so the on-disk
        # record matches what the API returns.
        result_path = self.output_dir / f"{slug}.json"
        attempt_result_path = attempt_dir / "result.json"
        _write_json_atomic(result_path, result, default=str)
        _write_json_atomic(attempt_result_path, result, default=str)
        if (
            self._online_score_executor is not None
            and result.get("outcome_status") == "pending_evaluation"
        ):
            self._online_score_executor.submit(
                score_saved_result,
                authoritative_result_path=attempt_result_path,
                latest_result_path=result_path,
            )

        self._ep_count += 1
        print(_summary(slug, result), flush=True)
        return result

    def _certification_result(
        self,
        *,
        episode: Any,
        result: dict[str, Any],
        runner: AutonomousEpisodeRunner,
        stage: str,
    ) -> dict[str, Any]:
        status_fn = getattr(self.agent, "certification_status", None)
        if not callable(status_fn):
            raise TypeError("Golden Oracle must expose certification_status().")
        from tarsphere.harness.certification.oracle import (
            episode_success_requires_retained_object,
        )

        status = dict(status_fn())
        completed = set(status.get("completed_gates") or [])
        # Compatibility key: "ever held" by a fixed joint or a physical hold.
        ever_attached = any(
            bool(heartbeat.get("held") or heartbeat.get("attached"))
            for heartbeat in runner.heartbeats
        )
        stability = runner.last_info.get("robot_stability")
        upright = bool(
            isinstance(stability, dict)
            and float(stability.get("pelvis_z_m", 0.0)) >= 0.45
            and float(stability.get("up_alignment", 0.0)) >= 0.45
        )
        healthy_completion = bool(
            "failure" not in result
            and result.get("termination_reason")
            in {
                "report_done",
                "diagnostic_checks",
            }
            and upright
        )
        grasp_evidence = self._grasp_certification_evidence(
            episode=episode,
            runner=runner,
            status=status,
        )
        gate_a = bool(
            healthy_completion
            and "A" in completed
            and ever_attached
            and grasp_evidence["passed"]
        )
        gate_b = bool(gate_a and "B" in completed)
        placement_evidence = None
        if stage == "C" and not episode_success_requires_retained_object(episode):
            placement_evidence = self._placement_release_evidence(runner)
        gate_c = bool(
            gate_b
            and "C" in completed
            and (
                bool(result.get("causal_graph_completed"))
                if episode_success_requires_retained_object(episode)
                else (placement_evidence is not None and placement_evidence.passed)
            )
        )
        pass_gates = {"A": gate_a, "B": gate_b, "C": gate_c}
        certification = {
            "certification_passed": bool(pass_gates[stage]),
            "pass_gates": pass_gates,
            "certification_status": status,
            "grasp_certification_evidence": grasp_evidence,
            "final_state": {
                "upright": upright,
                "ever_attached": ever_attached,
                "attached": bool(
                    runner.heartbeats and runner.heartbeats[-1].get("attached", False)
                ),
                "xy": (runner.heartbeats[-1].get("xy") if runner.heartbeats else None),
                "yaw_rad": (
                    runner.heartbeats[-1].get("yaw_rad") if runner.heartbeats else None
                ),
            },
        }
        if placement_evidence is not None:
            certification["placement_release_evidence"] = placement_evidence.model_dump(
                mode="json"
            )
        return certification

    @staticmethod
    def _grasp_certification_evidence(
        *,
        episode: Any,
        runner: AutonomousEpisodeRunner,
        status: dict[str, Any],
    ) -> dict[str, Any]:
        """Prove the declared enclosed grasp pose and carry-away."""
        world = getattr(episode, "world_knowledge", None) or {}
        grasp_motion_trial = world.get("grasp_motion_trial")
        grasp_system_trial = world.get("grasp_system_trial")
        physical_trial = world.get("physical_trial")
        physical_task = world.get("physical_task")
        if isinstance(grasp_motion_trial, dict):
            target_id = str(grasp_motion_trial.get("target_object_id") or "")
        elif isinstance(grasp_system_trial, dict):
            target_id = str(
                (grasp_system_trial.get("reference_object") or {}).get("object_id")
                or ""
            )
        elif isinstance(physical_trial, dict):
            target_id = str(
                (physical_trial.get("grasp") or {}).get("target_object_id") or ""
            )
        elif isinstance(physical_task, dict):
            target_id = str(physical_task.get("target_object_id") or "")
        else:
            target_id = ""
        if not target_id:
            raise RuntimeError(
                "Certification episode does not identify one grasp target."
            )

        reaches = [
            dict(item)
            for item in (status.get("grasp_reach_evidence") or [])
            if str(item.get("target_object_id") or "") == target_id
        ]
        minimum_aabb_gap = min(
            (float(item["aabb_surface_gap_m"]) for item in reaches),
            default=None,
        )
        displacement_threshold = G1_GRASP_CERTIFICATION_DISPLACEMENT_M
        maximum_attached_displacement = None
        carried_state = None
        attachment_evidence = None
        for step in runner.collector.steps:
            manipulation = step.manipulation
            objects = (
                manipulation.get("objects") if isinstance(manipulation, dict) else None
            )
            state = objects.get(target_id) if isinstance(objects, dict) else None
            if not isinstance(state, dict):
                continue
            live_attachment = state.get("grasp_attachment_evidence")
            if isinstance(live_attachment, dict):
                attachment_evidence = dict(live_attachment)
            position = state.get("position")
            initial = state.get("initial_position")
            shared_geometry_is_valid = (
                isinstance(position, (list, tuple))
                and isinstance(initial, (list, tuple))
                and len(position) == 3
                and len(initial) == 3
            )
            if not shared_geometry_is_valid:
                continue
            fixed_hold = bool(
                state.get("attached") is True
                and state.get("attachment_joint_valid") is True
                and state.get("hand") in {"left", "right"}
            )
            physical_hold = bool(
                state.get("physically_held") is True
                and state.get("held_by_hand") in {"left", "right"}
            )
            if not (fixed_hold or physical_hold):
                continue
            held_hand = str(state["hand"]) if fixed_hold else str(state["held_by_hand"])
            displacement = math.dist(
                [float(value) for value in position],
                [float(value) for value in initial],
            )
            if (
                maximum_attached_displacement is None
                or displacement > maximum_attached_displacement
            ):
                maximum_attached_displacement = displacement
                carried_state = {
                    "step": int(step.step),
                    "hand": held_hand,
                    "position_xyz": [float(value) for value in position],
                    "initial_position_xyz": [float(value) for value in initial],
                }

        if attachment_evidence is not None:
            from tarsphere.bench.certification.grasp_reach import (
                PhysXGraspContactEvidence,
            )

            finger_contacts = [
                PhysXGraspContactEvidence.from_recorded_contact(item).model_dump(
                    mode="python",
                )
                for item in (attachment_evidence.get("physx_contacts") or ())
                if isinstance(item, dict)
            ]
            attachment_evidence = {
                "hand": str(attachment_evidence.get("hand") or ""),
                "target_object_id": target_id,
                "finger_joint_positions": list(
                    attachment_evidence.get("finger_joint_positions") or ()
                ),
                "finger_closure_l2_rad": float(
                    attachment_evidence.get("finger_closure_l2_rad")
                ),
                "minimum_finger_closure_l2_rad": float(
                    attachment_evidence.get("minimum_finger_closure_l2_rad")
                ),
                "grasp_center_to_object_m": float(
                    attachment_evidence.get("grasp_center_to_object_m")
                ),
                "aabb_surface_gap_m": float(
                    attachment_evidence.get("aabb_surface_gap_m")
                ),
                "finger_contacts": finger_contacts,
            }

        grasp_pose_established = attachment_evidence is not None
        carried_away = bool(
            maximum_attached_displacement is not None
            and maximum_attached_displacement >= displacement_threshold
        )
        return {
            "target_object_id": target_id,
            "grasp_pose_established": grasp_pose_established,
            "attachment_evidence": attachment_evidence,
            "minimum_aabb_surface_gap_m": minimum_aabb_gap,
            "carried_away": carried_away,
            "maximum_attached_displacement_m": maximum_attached_displacement,
            "required_displacement_m": displacement_threshold,
            "carried_state": carried_state,
            "passed": bool(grasp_pose_established and carried_away),
        }

    def _placement_release_evidence(
        self,
        runner: AutonomousEpisodeRunner,
    ):
        """Measure Stage C from the diagnostic physical-task contract."""
        from tarsphere.bench.certification.placement_release import (
            measure_placement_release,
        )

        collector = runner.collector
        if collector is None or not getattr(collector, "steps", None):
            raise RuntimeError(
                "Stage C certification requires live manipulation snapshots"
            )
        world = runner.episode.world_knowledge
        physical_task = world.get("physical_task") if isinstance(world, dict) else None
        if not isinstance(physical_task, dict):
            raise RuntimeError(
                "Stage C certification requires world_knowledge.physical_task"
            )
        target_object_id = str(physical_task.get("target_object_id") or "").strip()
        destination_object_id = str(
            physical_task.get("final_destination_object_id") or ""
        ).strip()
        if not target_object_id or not destination_object_id:
            raise RuntimeError(
                "Stage C physical task must identify target and final destination"
            )
        return measure_placement_release(
            collector.steps,
            target_object_id=target_object_id,
            destination_object_id=destination_object_id,
            goal_relation=str(physical_task.get("goal_relation") or "on"),
        )

    @staticmethod
    def _git_sha() -> str | None:
        repo_root = Path(__file__).resolve().parents[3]
        # Never discover a parent repository: an isolated source snapshot can
        # live inside a checkout without inheriting that checkout's identity.
        if not (repo_root / ".git").exists():
            return None
        from dulwich.repo import Repo

        with Repo(repo_root) as repository:
            try:
                return repository.head().decode("ascii")
            except KeyError:
                return None

    @staticmethod
    def _source_tree_sha256() -> str:
        """Hash portable runtime source bytes, independently of Git metadata.

        ``HEAD`` alone is not an executable identity during development: the
        current harness migration contains both modified tracked files and new
        untracked source files. Hash relative paths plus bytes for every
        selected file under the two executable source roots. Episode YAML
        is sent in each request and recorded separately, so it is deliberately
        outside this code fingerprint.
        """

        repo_root = Path(__file__).resolve().parents[3]
        return runtime_source_tree_sha256(repo_root)

    @staticmethod
    def _worktree_dirty() -> bool | None:
        """Report whether executable source differs from the checked-out SHA."""

        repo_root = Path(__file__).resolve().parents[3]
        if not (repo_root / ".git").exists():
            return None
        completed = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "tarsphere",
                "scripts",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Unable to inspect runtime source status: "
                + (completed.stderr or "").strip()
            )
        return bool(completed.stdout.strip())

    def _hold_standing_robot(self, *, reason: str, settle_steps: int = 8) -> None:
        """Restore the saved standing articulation and step zero loco briefly.

        Persistent scene servers leave the Kit timeline playing between bake and
        ``/run_episode``. Without ``move_by_speed`` the G1 policy is not fed and
        gravity tips the body. Soft-reset recovery depends on a good standing
        snapshot; this hold both re-applies it and keeps the policy alive for a
        few steps so the next soft reset does not start from a fallen pose.
        """
        if self.env is None:
            return
        restore = getattr(self.env, "_restore_robot_episode_reset_state", None)
        intern_env = getattr(self.env, "_env", None)
        if not callable(restore) or intern_env is None:
            return
        attachment = getattr(self.env, "_g1_attachment", None)
        restore_objects = getattr(attachment, "restore_initial_state", None)
        if callable(restore_objects):
            restored = restore_objects()
            if restored:
                print(
                    f"[SERVER] Restored attached objects before standing hold: "
                    f"{restored}",
                    flush=True,
                )
        try:
            restore()
        except RuntimeError as exc:
            # First bake may call hold before the post-warmup snapshot exists.
            print(f"[SERVER] Standing hold skipped ({reason}): {exc}", flush=True)
            return
        # A failed episode may leave the articulation root physically prone.
        # Controller/joint restoration alone cannot lift that root.  This is a
        # between-episode reset operation, so restore the current episode's
        # authored spawn before advancing any physics tick.
        teleport = getattr(self.env, "_teleport_robot", None)
        episode = getattr(self.env, "episode", None)
        robot = getattr(episode, "robot", None)
        if callable(teleport) and robot is not None:
            teleport(
                [float(value) for value in robot.start_position],
                (
                    [float(value) for value in robot.start_orientation]
                    if robot.start_orientation is not None
                    else None
                ),
            )
        zero = {"move_by_speed": [0.0, 0.0, 0.0]}
        for _ in range(max(1, int(settle_steps))):
            obs, _, _, _, _ = intern_env.step(action=zero)
            self.env._obs = obs
            self.env._physics_steps = int(getattr(self.env, "_physics_steps", 0)) + 1
        assert_upright = getattr(self.env, "_assert_robot_upright", None)
        if callable(assert_upright):
            assert_upright(action_name=reason, env_action=zero)
        print(
            f"[SERVER] G1 standing hold ({reason}, steps={settle_steps}).", flush=True
        )

    def _warmup_agent_if_requested(self) -> None:
        if (
            self.agent is None
            or self.env is None
            or not heavy_backend_warmup_requested()
        ):
            return
        observation = self.env.observe()
        summary = warmup_configured_heavy_backends(self.agent, observation)
        print(f"[SERVER] Heavy backend warmup: {summary}", flush=True)

    def _prepare_episode_outputs(self, slug: str) -> Path:
        ep_image_dir = self.image_root / slug
        if ep_image_dir.exists():
            if not ep_image_dir.is_dir():
                raise RuntimeError(
                    f"Episode image path is not a directory: {ep_image_dir}"
                )
            shutil.rmtree(ep_image_dir)
        ep_image_dir.mkdir(parents=True, exist_ok=False)

        for stale_path in (
            self.video_dir / f"{slug}.mp4",
            self.video_dir / f"{slug}_ego.mp4",
            self.video_dir / f"{slug}_ego.ffmpeg.log",
            self.video_dir / f"{slug}_bev_trajectory.mp4",
            self.video_dir / f"{slug}_bev_trajectory.ffmpeg.log",
            self.video_dir / f"{slug}_bev_latest.png",
            self.video_dir / f"{slug}_ego_frames.jsonl",
            self.video_dir / f"{slug}_ego_with_bev.mp4",
            self.video_dir / f"{slug}_ego_model_output.ass",
            self.video_dir / f"{slug}_ego_with_bev_model_output.mp4",
            self.video_dir / f"{slug}_online_map.mp4",
            self.video_dir / f"{slug}_ego_decision_panel.mp4",
            self.output_dir / f"{slug}.json",
            self.output_dir / f"{slug}.evaluation_trajectory.json",
            self.output_dir / f"{slug}.heartbeat.jsonl",
            *(
                self.video_dir / f"{slug}_{view_name}.mp4"
                for view_name in ROOM_OBSERVER_VIEW_NAMES
            ),
            *(
                self.video_dir / f"{slug}_{view_name}_model_output.ass"
                for view_name in ROOM_OBSERVER_VIEW_NAMES
            ),
            *(
                self.video_dir / f"{slug}_{view_name}_model_output.mp4"
                for view_name in ROOM_OBSERVER_VIEW_NAMES
            ),
        ):
            if stale_path.exists():
                if not stale_path.is_file():
                    raise RuntimeError(
                        f"Stale episode output path is not a file: {stale_path}"
                    )
                stale_path.unlink()
        return ep_image_dir

    def _render_video(self, slug: str, image_dir: Path) -> Path:
        """Assemble decision frames without exposing private depth in RGB runs."""
        out = self.video_dir / f"{slug}.mp4"
        pattern = str(image_dir / "step_*_rgbd.png")
        frame_count = len(list(image_dir.glob("step_*_rgbd.png")))
        if frame_count == 0:
            raise RuntimeError(f"No RGBD frames found for video render: {image_dir}")
        filters = [f"fps={self.video_fps}"]
        filters.append("crop=iw/2:ih:0:0")
        filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(self.video_fps),
            "-pattern_type",
            "glob",
            "-i",
            pattern,
            "-vf",
            ",".join(filters),
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or "")[-400:]
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {tail}")
        return out

    def _render_continuous_ego_videos(
        self,
        *,
        slug: str,
        artifacts: ContinuousEgoVideoArtifacts,
        decision_trace: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        heartbeats: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    ) -> dict[str, str]:
        """Compose the original ego/map overlay with refined dark styling."""
        from tarsphere.infra.perception.ego_presentation import render_ego_presentation

        return render_ego_presentation(
            output_dir=self.output_dir,
            slug=slug,
            artifacts=artifacts,
            decision_trace=decision_trace,
            heartbeats=heartbeats,
        )

    def _render_room_observer_videos(
        self,
        slug: str,
        image_dir: Path,
    ) -> dict[str, str]:
        videos: dict[str, str] = {}
        observer_root = image_dir / "room_observer"
        # Room-observer frames are scheduled by physics tick in every render
        # mode. Encode at that same clock so formal evaluation rollouts do not
        # play a 5 Hz overhead trace at the unrelated decision-video rate.
        observer_fps = float(SIMULATION_PHYSICS_HZ) / float(self.room_observer_stride)
        for view_name in self.room_observer_view_names:
            frame_dir = observer_root / view_name
            frame_paths = sorted(frame_dir.glob("step_*.png"))
            frame_count = len(frame_paths)
            if frame_count == 0:
                raise RuntimeError(
                    f"No room-observer frames found for {view_name}: {frame_dir}"
                )
            out = self.video_dir / f"{slug}_{view_name}.mp4"
            clean_filters = [
                f"fps={observer_fps:.6g}",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            ]
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                f"{observer_fps:.6g}",
                "-pattern_type",
                "glob",
                "-i",
                str(frame_dir / "step_*.png"),
                "-vf",
                ",".join(clean_filters),
                "-pix_fmt",
                "yuv420p",
                str(out),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out.is_file():
                tail = (proc.stderr or "")[-400:]
                raise RuntimeError(
                    f"ffmpeg room observer {view_name} exited {proc.returncode}: {tail}"
                )
            videos[view_name] = str(out)
        return videos

    def _build_collector(self, episode: Any, env: Any) -> Any:
        """Construct the required trace collector without reading a stale frame.

        The environment is reset only after the runner starts.  Calling
        ``env.observe()`` here reads the previous episode's post-failure
        renderer state and can intermittently see a black frame.  Every valid
        decision observation already carries its own live camera parameters,
        so the collector learns them at ``record_step`` time instead.
        """
        del env
        from tarsphere.evaluation.private_trace import PrivateTraceCollector

        collector = PrivateTraceCollector()
        volumes = _load_object_volumes(episode)
        if volumes:
            collector.record_object_volumes(volumes)
        return collector

    def close(self, *, skip_kit_cleanup: bool = False) -> None:
        """Release the Isaac environment.

        Pass ``skip_kit_cleanup`` only from the process's final teardown: it
        leaves Kit unable to start another SimulationApp, and this class builds
        a fresh ``IsaacAutonomousEnv`` whenever it serves a new scene.
        """
        if self.env is not None:
            self.env.close(skip_kit_cleanup=skip_kit_cleanup)
            self.env = None
        if self._online_score_executor is not None:
            self._online_score_executor.shutdown(wait=True)
            self._online_score_executor = None
