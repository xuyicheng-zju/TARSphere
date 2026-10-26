#!/usr/bin/env python3
"""Evaluate supplied episodes with the default mechanism on one server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
from tarsphere.infra.process_lifecycle import (
    control_directory,
    exclusive_lock,
    host_state_directory,
    require_parent_lifetime,
)
from tarsphere.evaluation.result_layout import result_directory

from tarsphere.cli.episode_inputs import DEFAULT_EPISODES, load_episode_inputs
from tarsphere.configs.secrets import DEFAULT_AGENT_MAX_TOKENS

DEFAULT_NAVDP_PORT_BASE = 19000
NAVDP_PORTS_PER_GPU = 8
DEFAULT_MAX_STARTING_WORKERS = 8
DEFAULT_MIN_FREE_MEMORY_MIB = 7000
DEFAULT_WORKER_STALL_TIMEOUT_SECONDS = 3600
MAX_EVALUATION_ATTEMPTS_PER_PASS = 3
ENDPOINT_RETRY_INITIAL_SECONDS = 60.0
ENDPOINT_RETRY_CAP_SECONDS = 900.0
INFRASTRUCTURE_RETRY_INITIAL_SECONDS = 30.0
ISAAC_READY_MARKER = "[WORKER] Isaac context primed"
EXTERNAL_SERVICE_TERMINATIONS = frozenset(
    {
        "model_service_retry_exhausted",
        "evaluation_support_service_retry_exhausted",
    }
)


@dataclass(frozen=True)
class ModelEndpoint:
    name: str
    model: str
    api_base: str
    api_key_env: str
    api_key_fallback_env: str | None = None
    api_key_fallback_2_env: str | None = None
    max_tokens: int = DEFAULT_AGENT_MAX_TOKENS

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ValueError(
                f"{self.name}: required API key environment variable "
                f"{self.api_key_env} is empty"
            )
        return value

    def api_keys(self) -> tuple[str, ...]:
        primary = self.api_key()
        fallbacks = (
            os.environ.get(name, "").strip()
            for name in (self.api_key_fallback_env, self.api_key_fallback_2_env)
            if name
        )
        return tuple(dict.fromkeys((primary, *(key for key in fallbacks if key))))


def evaluation_support_binding(config: Any) -> dict[str, Any]:
    """Return the canonical semantic-evaluator identity saved with a run."""

    def field(name: str) -> Any:
        return config.get(name) if isinstance(config, dict) else getattr(config, name)

    return {
        "model": str(field("model") or "").strip(),
        "base_url": str(field("base_url") or "").strip().rstrip("/"),
        "timeout_seconds": float(field("timeout_seconds")),
    }


@dataclass(frozen=True)
class EpisodeRow:
    episode_id: str
    path: str
    sha256: str
    scene_asset_path: str
    scene_name: str
    mainline: str
    capability_slice: str


@dataclass
class SceneRun:
    """A subset of models evaluated while one scene remains resident."""

    job_id: str
    scene_asset_path: str
    episodes: list[EpisodeRow]
    models: list[ModelEndpoint]


@dataclass
class EvaluationJob:
    """One Isaac worker lane containing one or two scene loads."""

    job_id: str
    scene_runs: list[SceneRun]
    state: str = "queued"
    launch_sequence: int = 0
    consecutive_infrastructure_failures: int = 0
    endpoint_deferrals: int = 0
    next_launch_at_monotonic: float = 0.0
    last_exit_code: int | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class WorkerSlot:
    gpu: int
    runtime_slot: int

    @property
    def name(self) -> str:
        return f"gpu{self.gpu}_slot{self.runtime_slot}"


@dataclass
class RunningWorker:
    process: subprocess.Popen[Any]
    job: EvaluationJob
    slot: WorkerSlot
    log_path: Path
    launch_path: Path
    log_file: Any
    navdp_port: int
    last_activity_monotonic: float
    last_activity_mtime_ns: int = 0
    stall_terminated: bool = False
    stall_terminated_at: float | None = None
    pending_episode_run_count_at_launch: int = 0
    primed: bool = False
    log_read_offset: int = 0
    log_tail: str = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def load_episodes(path: Path, episode_ids: list[str] | None = None) -> list[EpisodeRow]:
    """Bind runtime episodes to their immutable source bytes."""
    return [
        EpisodeRow(
            episode_id=episode.episode_id,
            path=str(source),
            sha256=sha256(source),
            scene_asset_path=episode.scene.asset_path,
            scene_name=Path(episode.scene.asset_path).parent.name,
            mainline=episode.evaluation.task_profile.mainline.value,
            capability_slice=episode.evaluation.task_profile.capability_slice.value,
        )
        for source, episode in load_episode_inputs(path, episode_ids)
    ]


def build_jobs(
    models: list[ModelEndpoint], episodes: list[EpisodeRow]
) -> list[EvaluationJob]:
    """Group supplied episodes by resident scene while retaining their source identity."""
    scenes: dict[str, list[EpisodeRow]] = defaultdict(list)
    for row in episodes:
        scenes[row.scene_asset_path].append(row)
    jobs = []
    for scene, rows in sorted(scenes.items()):
        scene_id = hashlib.sha256(scene.encode()).hexdigest()[:12]
        jobs.append(
            EvaluationJob(
                job_id=f"scene_{scene_id}",
                scene_runs=[
                    SceneRun(
                        scene_id,
                        scene,
                        sorted(rows, key=lambda row: row.episode_id),
                        models,
                    )
                ],
            )
        )
    return jobs


def evaluation_slots(
    gpu_count: int,
    *,
    reserve_gpu0_for_services: bool = False,
    workers_per_gpu: tuple[int, ...] | None = None,
) -> list[WorkerSlot]:
    if gpu_count < 1:
        raise ValueError("At least one NVIDIA GPU is required")
    if workers_per_gpu is not None and reserve_gpu0_for_services:
        raise ValueError("--workers-per-gpu and --reserve-gpu0-for-services conflict")
    counts = workers_per_gpu or tuple(
        0 if gpu == 0 and reserve_gpu0_for_services else 1 for gpu in range(gpu_count)
    )
    if len(counts) != gpu_count or any(not 0 <= count <= 4 for count in counts):
        raise ValueError(
            f"--workers-per-gpu requires {gpu_count} counts between 0 and 4"
        )
    if not any(counts):
        raise ValueError("At least one Isaac worker slot is required")
    # Fill one slot on each GPU before a second cold RTX/material build.
    return [
        WorkerSlot(gpu, runtime_slot)
        for runtime_slot in range(max(counts))
        for gpu, count in enumerate(counts)
        if runtime_slot < count
    ]


def parse_workers_per_gpu(value: str) -> tuple[int, ...]:
    """Map worker counts to the GPUs in CUDA visibility order."""

    try:
        counts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--workers-per-gpu must contain comma-separated integers"
        ) from error
    if not counts or any(not 0 <= count <= 4 for count in counts):
        raise argparse.ArgumentTypeError(
            "--workers-per-gpu requires counts between 0 and 4"
        )
    return counts


def detected_gpu_count() -> int:
    from tarsphere.cli.gpu_selection import visible_gpu_indices

    return len(visible_gpu_indices())


def free_gpu_memory() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    )
    return {
        int(line.split(",")[0].strip()): int(line.split(",")[1].strip())
        for line in output.splitlines()
    }


def physical_cpu_cores() -> int:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cores: set[tuple[str, str]] = set()
        for block in cpuinfo.read_text(encoding="utf-8").split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            if "physical id" in fields and "core id" in fields:
                cores.add((fields["physical id"], fields["core id"]))
        if cores:
            return len(cores)
    return os.cpu_count() or 1


def check_agent_endpoint(model: ModelEndpoint) -> None:
    """Probe the generation API used by the worker, including key fallbacks."""
    failures: list[str] = []
    for credential_index, api_key in enumerate(model.api_keys(), start=1):
        try:
            # OpenAI-compatible generation does not require a model-list API.
            # Use the same SDK/transport as the worker: a separate urllib TLS
            # stack or an unavailable /models route must not block valid chat.
            _check_chat_inference(
                name=model.name,
                model=model.model,
                api_base=model.api_base,
                api_key=api_key,
            )
            return
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            failures.append(
                f"credential {credential_index}: "
                f"{type(exc).__name__}"
                + (f" HTTP {status}" if status is not None else "")
            )
    raise RuntimeError(
        f"{model.name} endpoint is not healthy at {model.api_base}; "
        + "; ".join(failures)
    )


def _check_chat_inference(
    *,
    name: str,
    model: str,
    api_base: str,
    api_key: str,
) -> None:
    """Require one real generation, not only a TCP or /models response."""

    try:
        from openai import OpenAI
        from tarsphere.harness.kernel.vlm_client import (
            LOCAL_EMBODIED_REPETITION_PENALTIES,
            LOCAL_EMBODIED_THINKING_KWARG_MODELS,
            PROVIDER_THINKING_IN_CONTENT_MODELS,
        )
        from tarsphere.provider_compat import NON_THINKING_EVALUATION_MODELS

        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly READY.",
                }
            ],
            # A reasoning-capable provider can spend a handful of tokens
            # before producing the requested visible word. Keep this probe
            # small, but large enough that empty content is a real failure.
            "max_tokens": 64,
        }
        extra_body: dict[str, Any] = {}
        if model in (
            PROVIDER_THINKING_IN_CONTENT_MODELS | LOCAL_EMBODIED_THINKING_KWARG_MODELS
        ):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if model in NON_THINKING_EVALUATION_MODELS:
            extra_body["enable_thinking"] = False
        repetition_penalty = LOCAL_EMBODIED_REPETITION_PENALTIES.get(model)
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty
        if extra_body:
            request["extra_body"] = extra_body
        client = OpenAI(
            api_key=api_key,
            base_url=api_base or None,
            timeout=120.0,
            max_retries=0,
        )
        try:
            response = client.chat.completions.create(**request)
        finally:
            client.close()
        content = str(response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("completion returned empty content")
    except Exception as exc:
        raise RuntimeError(
            f"{name} failed the startup inference probe for model "
            f"{model!r} at {api_base}: {exc}"
        ) from exc


def check_local_model_services() -> None:
    from tarsphere.harness.perception.remote_backends import SAM3ServiceClient
    from tarsphere.infra.manipulation.graspgenx import (
        GraspGenXClient,
        GraspGenXSettings,
    )

    SAM3ServiceClient(timeout_seconds=10).request({"cmd": "sam3_require_ready"})
    grasp = GraspGenXClient(
        settings=GraspGenXSettings(
            host=os.environ.get("TARSPHERE_GRASPGENX_HOST", "127.0.0.1"),
            port=int(os.environ.get("TARSPHERE_GRASPGENX_PORT", "5556")),
        )
    )
    try:
        grasp.require_ready()
    finally:
        grasp.close()


def navdp_port(slot: WorkerSlot, base: int) -> int:
    if not 0 <= slot.runtime_slot < NAVDP_PORTS_PER_GPU:
        raise ValueError(f"Invalid X-NavDP runtime slot: {slot.runtime_slot}")
    port = base + slot.gpu * NAVDP_PORTS_PER_GPU + slot.runtime_slot
    if not 1 <= port <= 65535:
        raise ValueError(f"Derived X-NavDP port is invalid: {port}")
    return port


def host_id() -> str:
    """Return the filesystem-safe short hostname used by service launchers."""

    short_name = socket.gethostname().split(".", 1)[0]
    value = "".join(
        character
        for character in short_name
        if character.isalnum() or character in "_.-"
    )
    if not value:
        raise RuntimeError("Could not derive a filesystem-safe host id")
    return value


def navdp_pid_file(port: int) -> Path:
    configured = os.environ.get("TARSPHERE_XNAVDP_STATE_DIR", "").strip()
    state_dir = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "tarsphere" / host_id() / "x_navdp"
    )
    return state_dir / f"x-navdp-{port}.pid"


def stop_navdp(port: int) -> None:
    path = navdp_pid_file(port)
    if not path.is_file():
        return
    raw = "".join(character for character in path.read_text() if character.isdigit())
    if not raw:
        path.unlink(missing_ok=True)
        return
    pid = int(raw)
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except (FileNotFoundError, ProcessLookupError):
        path.unlink(missing_ok=True)
        return
    if (
        b"eval.src.policy_server" not in command
        or f"--port {port}".encode() not in command
    ):
        raise RuntimeError(f"Refusing to stop PID {pid}; it is not X-NavDP port {port}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"X-NavDP PID {pid} did not stop within 10 seconds")
    path.unlink(missing_ok=True)


def start_navdp(slot: WorkerSlot, port: int) -> None:
    stop_navdp(port)
    environment = dict(os.environ)
    environment.update(
        {
            "TARSPHERE_MODEL_GPU": str(slot.gpu),
            "TARSPHERE_XNAVDP_PORT": str(port),
        }
    )
    result = subprocess.run(
        ["bash", "tarsphere/infra/services/navdp.sh", "--new-server"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"X-NavDP failed for {slot.name} on port {port}:\n"
            f"{result.stdout}{result.stderr}"
        )


def safe_name(value: str) -> str:
    return (
        "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in value
        ).strip("_")
        or "item"
    )


def model_output_dir(output: Path, scene: SceneRun, model: ModelEndpoint) -> Path:
    names = {row.scene_name for row in scene.episodes}
    if len(names) != 1:
        raise ValueError(f"Conflicting scene names in {scene.job_id}: {names}")
    return output / model.name / next(iter(names))


def result_path(
    output: Path, scene: SceneRun, model: ModelEndpoint, episode: EpisodeRow
) -> Path:
    return (
        result_directory(
            model_output_dir(output, scene, model),
            episode.mainline,
            episode.capability_slice,
        )
        / "result.json"
    )


def _validated_result_state(
    path: Path,
    row: EpisodeRow,
    *,
    scene: SceneRun,
    model: ModelEndpoint,
) -> str:
    """Classify a result for resume from its bound, typed score.

    Older company-host results may arrive before their large rollout and input
    archives. A valid saved score prevents paying for the same physical run
    twice; evidence completeness is checked separately after the archives
    finish merging. An infrastructure failure never becomes an Agent score.
    """

    if not path.is_file():
        return "physical_pending"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "physical_pending"
    if not isinstance(payload, dict) or payload.get("episode_id") != row.episode_id:
        return "physical_pending"
    if payload.get("evaluation_identity") != {
        "schema_version": "tarsphere.result_identity.v1",
        "episode_id": row.episode_id,
        "episode_sha256": row.sha256,
        "scene_job_id": scene.job_id,
        "model_name": model.name,
        "model_id": model.model,
    }:
        return "physical_pending"
    # A provider transport outage is infrastructure failure, not an Agent
    # outcome.  Keep the durable artifact for diagnosis, but make resume run
    # the physical episode again once the endpoint is healthy.
    if payload.get("termination_reason") in EXTERNAL_SERVICE_TERMINATIONS:
        return "physical_pending"
    status = payload.get("outcome_status")
    from tarsphere.evaluation.online_scoring import has_archived_binary_score

    if status in {
        "scored_pending_attribution",
        "evaluated",
    } and has_archived_binary_score(payload):
        # A scoring-version change never schedules another physical rollout.
        # The scorer validates frozen inputs before calling a judge.
        return "evaluation_pending"
    if status in {"scored_pending_attribution", "evaluated"}:
        try:
            from tarsphere.evaluation.schema import (
                EpisodeEvaluationReport,
                OnlineEpisodeScore,
            )

            score = OnlineEpisodeScore.model_validate(payload["online_score"])
            if (
                score.episode_id != row.episode_id
                or score.task_profile.mainline.value != row.mainline
                or score.task_profile.capability_slice.value != row.capability_slice
                or score.termination_reason != payload.get("termination_reason")
                or score.task_progress.success is not payload.get("success")
            ):
                return "physical_pending"
            if status == "scored_pending_attribution":
                return "evaluation_pending"
            report = EpisodeEvaluationReport.model_validate(
                payload["evaluation_report"]
            )
            if (
                report.episode_id != score.episode_id
                or report.task_profile != score.task_profile
                or report.task_progress != score.task_progress
                or report.decision_steps != score.decision_steps
                or report.action_steps != score.action_steps
                or report.physics_steps != score.physics_steps
                or report.termination_reason != score.termination_reason
                or (
                    not report.task_progress.success
                    and not report.failure_attribution.analyzed
                )
            ):
                return "physical_pending"
            return "evaluated"
        except (ImportError, KeyError, ValueError, TypeError):
            return "physical_pending"
    for artifact_key in (
        "episode_artifact",
        "rollout_artifact",
        "evaluation_trajectory",
        "heartbeat",
        "model_output_trace",
    ):
        artifact_path_value = payload.get(f"{artifact_key}_path")
        artifact_hash = payload.get(f"{artifact_key}_sha256")
        if not isinstance(artifact_path_value, str) or not isinstance(
            artifact_hash, str
        ):
            return "physical_pending"
        artifact_path = Path(artifact_path_value)
        try:
            if (
                not artifact_path.is_file()
                or not artifact_path.resolve().is_relative_to(path.parent.resolve())
                or sha256(artifact_path) != artifact_hash
            ):
                return "physical_pending"
        except OSError:
            return "physical_pending"
    # Results created before the explicit flag retain the release's original
    # video-required contract. New throughput-first runs set it to false.
    if payload.get("video_recording_enabled", True):
        video = payload.get("video")
        continuous_videos = payload.get("continuous_ego_videos")
        continuous_hashes = payload.get("continuous_ego_video_sha256")
        if not all(
            isinstance(value, dict)
            for value in (
                video,
                continuous_videos,
                continuous_hashes,
            )
        ):
            return "physical_pending"
        assert isinstance(video, dict)
        assert isinstance(continuous_videos, dict)
        assert isinstance(continuous_hashes, dict)
        required_video_paths = {
            "ego": (video.get("path"), video.get("sha256")),
            "frame_manifest": (
                video.get("manifest_path"),
                video.get("manifest_sha256"),
            ),
            **{
                key: (continuous_videos.get(key), continuous_hashes.get(key))
                for key in (
                    "online_map",
                    "model_output_ass",
                    "ego_decision_panel",
                )
            },
        }
        for artifact_path_value, artifact_hash in required_video_paths.values():
            if not isinstance(artifact_path_value, str) or not isinstance(
                artifact_hash, str
            ):
                return "physical_pending"
            artifact_path = Path(artifact_path_value)
            try:
                if (
                    not artifact_path.is_file()
                    or not artifact_path.resolve().is_relative_to(path.parent.resolve())
                    or sha256(artifact_path) != artifact_hash
                ):
                    return "physical_pending"
            except OSError:
                return "physical_pending"
    try:
        from tarsphere.evaluation.official import validate_formal_trajectory_budget
        from tarsphere.evaluation.schema import EpisodeTrajectory
        from tarsphere.harness.kernel.model_input_snapshot import (
            validate_model_input_trace,
        )
        from tarsphere.bench.schema import TARSphereEpisode

        frozen_episode = TARSphereEpisode.model_validate_json(
            Path(payload["episode_artifact_path"]).read_text(encoding="utf-8")
        )
        trajectory = EpisodeTrajectory.model_validate_json(
            Path(payload["evaluation_trajectory_path"]).read_text(encoding="utf-8")
        )
        validate_formal_trajectory_budget(trajectory)
        validate_model_input_trace(
            Path(payload["model_output_trace_path"]),
            trajectory=trajectory,
            artifact_root=path.parent,
            model_id=model.model,
        )
    except (ImportError, KeyError, OSError, UnicodeDecodeError, ValueError, TypeError):
        return "physical_pending"
    if (
        trajectory.episode_id != row.episode_id
        or frozen_episode.episode_id != row.episode_id
        or frozen_episode.evaluation is None
        or frozen_episode.evaluation.task_profile != trajectory.task_profile
    ):
        return "physical_pending"
    if status in {
        "pending_evaluation",
        "evaluation_failed",
        "scored_pending_attribution",
    }:
        return "evaluation_pending"
    if status != "evaluated":
        return "physical_pending"
    try:
        from tarsphere.evaluation.schema import EpisodeEvaluationReport

        report = EpisodeEvaluationReport.model_validate(
            payload.get("evaluation_report")
        )
    except (ImportError, ValueError, TypeError):
        return "physical_pending"
    if (
        report.episode_id != row.episode_id
        or report.task_profile != trajectory.task_profile
        or report.decision_steps != trajectory.decision_steps
        or report.action_steps != len(trajectory.steps)
        or report.physics_steps != trajectory.steps[-1].physics_step
        or report.termination_reason != trajectory.termination_reason
        or payload.get("success") is not report.task_progress.success
    ):
        return "physical_pending"
    if not report.task_progress.success and not report.failure_attribution.analyzed:
        # A failed online score is physically complete but is not a final
        # evaluation until the offline attribution pass has run.
        return "evaluation_pending"
    return "evaluated"


# This cache is only for live progress polling. Resume, schedule construction,
# offline scoring and the final completion gate always validate full artifacts.
_monitor_cache: dict[tuple, tuple[tuple[int, ...], str]] | None = None


def _monitored_result_state(
    path: Path, row: EpisodeRow, *, scene: SceneRun, model: ModelEndpoint
) -> str:
    if _monitor_cache is None:
        return _validated_result_state(path, row, scene=scene, model=model)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return "physical_pending"
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = (str(path), row, scene.job_id, model)
    previous = _monitor_cache.get(key)
    if previous is not None and previous[0] == signature:
        return previous[1]
    state = _validated_result_state(path, row, scene=scene, model=model)
    _monitor_cache[key] = (signature, state)
    return state


def pending_episodes(
    output: Path, scene: SceneRun, model: ModelEndpoint
) -> list[EpisodeRow]:
    return [
        row
        for row in scene.episodes
        if _monitored_result_state(
            result_path(output, scene, model, row),
            row,
            scene=scene,
            model=model,
        )
        == "physical_pending"
    ]


def evaluation_pending_episodes(
    output: Path, scene: SceneRun, model: ModelEndpoint
) -> list[EpisodeRow]:
    return [
        row
        for row in scene.episodes
        if _monitored_result_state(
            result_path(output, scene, model, row),
            row,
            scene=scene,
            model=model,
        )
        == "evaluation_pending"
    ]


def pending_episode_count(output: Path, job: EvaluationJob) -> int:
    return sum(
        len(pending_episodes(output, scene, model))
        for scene in job.scene_runs
        for model in scene.models
    )


def unfinished_episode_count(output: Path, job: EvaluationJob) -> int:
    return sum(
        _monitored_result_state(
            result_path(output, scene, model, row),
            row,
            scene=scene,
            model=model,
        )
        != "evaluated"
        for scene in job.scene_runs
        for model in scene.models
        for row in scene.episodes
    )


def evaluation_pending_count(output: Path, job: EvaluationJob) -> int:
    return sum(
        len(evaluation_pending_episodes(output, scene, model))
        for scene in job.scene_runs
        for model in scene.models
    )


def retry_pending_evaluations(output: Path, jobs: list[EvaluationJob]) -> None:
    """Retry Judge/failure attribution from saved trajectories only."""

    from tarsphere.api_usage import aggregate_model_calls, merge_usage_reports
    from tarsphere.bench.schema import TARSphereEpisode
    from tarsphere.evaluation.official import (
        EvaluationRunError,
        apply_official_evaluation_outcome,
        evaluate_official_trajectory,
    )
    from tarsphere.evaluation.schema import (
        EpisodeTrajectory,
        OnlineEpisodeScore,
        TARSPHERE_SCORING_VERSION,
    )
    from tarsphere.evaluation.online_scoring import archive_previous_evaluation

    for job in jobs:
        for scene in job.scene_runs:
            for model in scene.models:
                for row in evaluation_pending_episodes(output, scene, model):
                    path = result_path(output, scene, model, row)
                    result = json.loads(path.read_text(encoding="utf-8"))
                    archive = archive_previous_evaluation(path)
                    if archive is not None:
                        result["previous_evaluation_archive"] = str(archive)
                    attempts = int(result.get("evaluation_attempts") or 0)
                    pass_limit = attempts + MAX_EVALUATION_ATTEMPTS_PER_PASS
                    try:
                        # Score the exact episode frozen with this physical
                        # attempt, not a mutable authoring YAML at retry time.
                        episode = TARSphereEpisode.model_validate_json(
                            Path(result["episode_artifact_path"]).read_text(
                                encoding="utf-8"
                            )
                        )
                        trajectory_path = Path(result["evaluation_trajectory_path"])
                        trajectory = EpisodeTrajectory.model_validate_json(
                            trajectory_path.read_text(encoding="utf-8")
                        )
                    except (
                        OSError,
                        UnicodeDecodeError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        result["outcome_status"] = "infrastructure_invalid"
                        result["success"] = False
                        result["evaluation_input_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        atomic_json(path, result)
                        continue
                    while attempts < pass_limit:
                        attempts += 1
                        result["evaluation_attempts"] = attempts
                        try:
                            report, usage = evaluate_official_trajectory(
                                episode, trajectory
                            )
                            result["evaluation_report"] = report.model_dump(mode="json")
                            result["online_score"] = OnlineEpisodeScore(
                                **report.model_dump(
                                    exclude={"schema_version", "failure_attribution"},
                                )
                            ).model_dump(mode="json")
                            result.setdefault("evaluation_attempt_history", []).append(
                                {
                                    "attempt": attempts,
                                    "scoring_version": TARSPHERE_SCORING_VERSION,
                                    "status": "evaluated",
                                    "usage": usage,
                                }
                            )
                            result.setdefault(
                                "evaluation_model_usage_attempts", []
                            ).append(usage)
                            result["evaluation_model_usage"] = merge_usage_reports(
                                result["evaluation_model_usage_attempts"]
                            )
                            apply_official_evaluation_outcome(result, report)
                            result["failure_attribution_status"] = (
                                "not_required"
                                if report.task_progress.success
                                else "completed"
                            )
                            result["api_usage"] = merge_usage_reports(
                                (
                                    result.get("agent_model_usage")
                                    or aggregate_model_calls([]),
                                    result.get("human_model_usage")
                                    or aggregate_model_calls([]),
                                    result["evaluation_model_usage"],
                                )
                            )
                            atomic_json(path, result)
                            break
                        except Exception as exc:  # preserve artifact for next retry
                            usage = (
                                exc.usage
                                if isinstance(exc, EvaluationRunError)
                                else aggregate_model_calls([])
                            )
                            result.pop("online_score", None)
                            result["evaluation_report"] = {"error": str(exc)}
                            result.setdefault("evaluation_attempt_history", []).append(
                                {
                                    "attempt": attempts,
                                    "scoring_version": TARSPHERE_SCORING_VERSION,
                                    "status": "evaluation_failed",
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "usage": usage,
                                }
                            )
                            result.setdefault(
                                "evaluation_model_usage_attempts", []
                            ).append(usage)
                            result["evaluation_model_usage"] = merge_usage_reports(
                                result["evaluation_model_usage_attempts"]
                            )
                            result["api_usage"] = merge_usage_reports(
                                (
                                    result.get("agent_model_usage")
                                    or aggregate_model_calls([]),
                                    result.get("human_model_usage")
                                    or aggregate_model_calls([]),
                                    result["evaluation_model_usage"],
                                )
                            )
                            result["success"] = False
                            result["outcome_status"] = "evaluation_failed"
                            atomic_json(path, result)


def scene_worker_dir(output: Path, job: EvaluationJob) -> Path:
    return control_directory(output) / "workers" / job.job_id


def model_scene_batch_count(job: EvaluationJob) -> int:
    return sum(len(scene.models) for scene in job.scene_runs)


def job_model_sequence(job: EvaluationJob) -> list[str]:
    return [model.name for scene in job.scene_runs for model in scene.models]


def schedule_document(
    output: Path, job: EvaluationJob, *, excluded_models: set[str] | None = None
) -> dict[str, Any]:
    scene_runs = []
    for scene in job.scene_runs:
        model_runs = []
        for model in scene.models:
            if model.name in (excluded_models or set()):
                continue
            pending = [
                row
                for row in scene.episodes
                if _validated_result_state(
                    result_path(output, scene, model, row),
                    row,
                    scene=scene,
                    model=model,
                )
                == "physical_pending"
            ]
            if not pending:
                continue
            destination = model_output_dir(output, scene, model)
            destination.mkdir(parents=True, exist_ok=True)
            model_runs.append(
                {
                    "name": model.name,
                    "model": model.model,
                    "api_base": model.api_base,
                    "max_tokens": model.max_tokens,
                    "api_key_env": model.api_key_env,
                    "api_key_fallback_env": model.api_key_fallback_env,
                    "api_key_fallback_2_env": model.api_key_fallback_2_env,
                    "output_dir": str(destination),
                    "episodes": [
                        {
                            "episode_id": row.episode_id,
                            "path": row.path,
                            "sha256": row.sha256,
                        }
                        for row in pending
                    ],
                }
            )
        if model_runs:
            scene_runs.append(
                {
                    "scene_job_id": scene.job_id,
                    "scene_asset_path": scene.scene_asset_path,
                    "model_runs": model_runs,
                }
            )
    return {
        "schema_version": "tarsphere.isaac_worker_schedule.v4",
        "job_id": job.job_id,
        "scene_runs": scene_runs,
    }


def worker_environment(
    slot: WorkerSlot, port: int, cache_root: Path, cpu_threads: int
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(
        {
            "PYTHONPATH": f"{ROOT}:{environment.get('PYTHONPATH', '')}",
            "TARSPHERE_KIT_ACTIVE_GPU": str(slot.gpu),
            "TARSPHERE_ISAAC_RUNTIME_SLOT": str(slot.runtime_slot),
            "TARSPHERE_MODEL_GPU": str(slot.gpu),
            # Isaac needs the full physical GPU list, while its isolated cuRobo
            # child must see only this lane's GPU. Without this binding every lane
            # contends on the cuRobo "default" lock and may plan on GPU0.
            "TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES": str(slot.gpu),
            "TARSPHERE_XNAVDP_PORT": str(port),
            "TARSPHERE_FRONTIERNET_DEVICE": f"cuda:{slot.gpu}",
            "OMP_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
            "PYTHONUNBUFFERED": "1",
            "TARSPHERE_SCHEDULER_PID": str(os.getpid()),
            "XDG_CACHE_HOME": str(cache_root / "xdg" / slot.name),
            "OMNI_USER_CACHE": str(cache_root / "ov" / slot.name),
        }
    )
    return environment


def worker_command(
    schedule_path: Path,
    *,
    evaluate_after_rollout: bool = True,
    harness_config=None,
) -> list[str]:
    command = [
        sys.executable,
        "tarsphere/cli/worker.py",
        "--evaluation-schedule",
        str(schedule_path),
        "--headless",
        "--no-navmesh",
        "--human-motion-backend",
        "kinematic",
        "--render-mode",
        "evaluation",
        "--viewer",
        "first_person",
        "--warmup-heavy-backends",
    ]
    if evaluate_after_rollout:
        command.append("--evaluate-after-rollout")
    if harness_config is None:
        from tarsphere.harness.harness import AgentHarnessConfig

        harness_config = AgentHarnessConfig()
    protocol = harness_config.protocol
    controller = harness_config.controller
    command.extend(
        [
            "--text-history-horizon",
            str(protocol.text_history_horizon),
            "--visual-history-horizon",
            str(protocol.visual_history_horizon),
            "--action-horizon",
            str(protocol.action_horizon),
            "--camera-views",
            ",".join(view.value for view in protocol.camera_views),
            "--online-mapping-hz",
            str(controller.online_mapping_hz),
        ]
    )
    return command


def validate_harness_output_config(
    output: Path,
    *,
    harness_config: Any,
) -> None:
    """Reject reuse of results produced under another experiment condition."""

    expected = harness_config.as_dict()
    for plan_path in sorted(control_directory(output).glob("host*/plan.json")):
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        recorded = previous.get("agent_harness_config")
        if recorded is not None and recorded != expected:
            raise ValueError(
                f"{plan_path} belongs to a different AgentHarnessConfig; "
                "use a separate --output-dir for this experiment condition"
            )
    # Plans written before the configuration field existed cannot establish
    # provenance. Their per-episode result records still carry the resolved
    # config, so check them before any scheduler resume decision is made.
    for result_file in output.rglob("result.json"):
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue  # A damaged result is already classified for rerun.
        if not isinstance(result, dict) or "evaluation_identity" not in result:
            continue
        if result.get("outcome_status") == "infrastructure_invalid":
            continue  # Never reused by the scheduler; it must rerun.
        if result.get("agent_harness_config") != expected:
            raise ValueError(
                f"{result_file} has a different or missing AgentHarnessConfig; "
                "use a separate --output-dir for this experiment condition"
            )


def worker_is_primed(worker: RunningWorker) -> bool:
    if worker.primed:
        return True
    if not worker.log_path.is_file():
        return False
    try:
        with worker.log_path.open("r", encoding="utf-8", errors="replace") as log:
            log.seek(worker.log_read_offset)
            text = worker.log_tail + log.read()
            worker.log_read_offset = log.tell()
        worker.primed = ISAAC_READY_MARKER in text
        worker.log_tail = text[-len(ISAAC_READY_MARKER) :]
        return worker.primed
    except OSError:
        return False


def worker_activity_mtime_ns(output: Path, job: EvaluationJob, log_path: Path) -> int:
    """Read only this job's log and assigned episode heartbeats.

    Other lanes can run the same scene/model, so a scene-wide heartbeat glob
    would hide a stalled worker behind their progress.
    """

    paths = [log_path]
    for scene in job.scene_runs:
        for model in scene.models:
            paths.extend(
                result_path(output, scene, model, episode).with_name(
                    "episode.heartbeat.jsonl"
                )
                for episode in scene.episodes
            )
    newest = 0
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime_ns)
        except OSError:
            continue
    return newest


def refresh_worker_activity(worker: RunningWorker, output: Path) -> None:
    """Advance the deadman clock from this worker's durable activity."""

    newest = worker_activity_mtime_ns(output, worker.job, worker.log_path)
    if newest > worker.last_activity_mtime_ns:
        worker.last_activity_mtime_ns = newest
        worker.last_activity_monotonic = time.monotonic()


def retry_delay_seconds(
    job_id: str,
    attempt: int,
    *,
    initial_seconds: float,
    cap_seconds: float,
) -> float:
    """Return deterministic exponential backoff with per-lane jitter."""

    if type(attempt) is not int or attempt < 1:
        raise ValueError("retry attempt must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (initial_seconds, cap_seconds)):
        raise ValueError("retry delays must be finite and positive")
    exponent = attempt - 1
    # Compare logs before exponentiation: a long outage must not overflow
    # while converting an enormous integer multiplier to a float.
    if exponent >= math.log2(cap_seconds) - math.log2(initial_seconds):
        base = cap_seconds
    else:
        base = math.ldexp(initial_seconds, exponent)
    digest = hashlib.sha256(f"{job_id}:{attempt}".encode()).digest()
    jitter_fraction = int.from_bytes(digest[:2], "big") / 65535.0
    return min(base + initial_seconds * 0.25 * jitter_fraction, cap_seconds)


def job_external_service_outages(output: Path, job: EvaluationJob) -> set[str]:
    """Return service keys whose pending rows exhausted transport recovery."""

    services: set[str] = set()
    for scene in job.scene_runs:
        for model in scene.models:
            for row in pending_episodes(output, scene, model):
                path = result_path(output, scene, model, row)
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                termination = payload.get("termination_reason")
                if termination == "model_service_retry_exhausted":
                    services.add(model.name)
                elif termination == "evaluation_support_service_retry_exhausted":
                    services.add("evaluation_support")
    return services


def job_has_external_service_outage(output: Path, job: EvaluationJob) -> bool:
    return bool(job_external_service_outages(output, job))


def make_job_retryable(job: EvaluationJob, *, delay_seconds: float) -> None:
    job.state = "cooldown"
    job.next_launch_at_monotonic = time.monotonic() + delay_seconds


def release_ready_jobs(jobs: list[EvaluationJob]) -> None:
    now = time.monotonic()
    for job in jobs:
        if job.state == "cooldown" and job.next_launch_at_monotonic <= now:
            job.state = "queued"


def job_record(job: EvaluationJob, output: Path) -> dict[str, Any]:
    total = sum(len(scene.models) * len(scene.episodes) for scene in job.scene_runs)
    physical_pending = pending_episode_count(output, job)
    unfinished = unfinished_episode_count(output, job)
    return {
        "job_id": job.job_id,
        "scene_count": len(job.scene_runs),
        "model_scene_batch_count": sum(len(scene.models) for scene in job.scene_runs),
        "completed_episode_run_count": total - unfinished,
        "pending_episode_run_count": unfinished,
        "physical_pending_episode_run_count": physical_pending,
        "evaluation_pending_episode_run_count": evaluation_pending_count(output, job),
        "state": job.state,
        "launch_sequence": job.launch_sequence,
        "consecutive_infrastructure_failures": (
            job.consecutive_infrastructure_failures
        ),
        "endpoint_deferrals": job.endpoint_deferrals,
        "retry_in_seconds": (
            max(0.0, job.next_launch_at_monotonic - time.monotonic())
            if job.state == "cooldown"
            else 0.0
        ),
        "last_exit_code": job.last_exit_code,
        "last_error": job.last_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from tarsphere.harness.harness_cli import add_agent_harness_arguments

    add_agent_harness_arguments(parser)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=DEFAULT_EPISODES,
        help="Episode YAML, directory of YAML files, or JSON path index.",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        help="Select an episode ID from the input; repeat for a subset.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TARSPHERE_AGENT_MODEL", ""),
        help="Remote vision model identifier; defaults to TARSPHERE_AGENT_MODEL.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_AGENT_MAX_TOKENS,
        help="Positive Agent completion token budget per decision (default: 16384).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("TARSPHERE_AGENT_API_BASE", ""),
        help="Agent provider API base URL; defaults to TARSPHERE_AGENT_API_BASE.",
    )
    parser.add_argument(
        "--api-key-env",
        default="TARSPHERE_AGENT_API_KEY",
        help="Environment variable containing the Agent API key.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path.home() / ".local/share/tarsphere/runs"
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--evaluate-after-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Queue online task scoring after each saved rollout while Isaac "
            "continues. Enabled by default; failure attribution remains a "
            "separate offline pass."
        ),
    )
    parser.add_argument("--navdp-port-base", type=int, default=DEFAULT_NAVDP_PORT_BASE)
    parser.add_argument(
        "--max-starting-workers", type=int, default=DEFAULT_MAX_STARTING_WORKERS
    )
    parser.add_argument(
        "--min-free-memory-mib", type=int, default=DEFAULT_MIN_FREE_MEMORY_MIB
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=parse_workers_per_gpu,
        help="Comma-separated Isaac worker counts in GPU index order; default one per GPU.",
    )
    parser.add_argument(
        "--reserve-gpu0-for-services",
        action="store_true",
        help="Reserve physical GPU 0 for shared SAM3 and GraspGenX services.",
    )
    parser.add_argument(
        "--worker-stall-timeout-seconds",
        type=int,
        default=DEFAULT_WORKER_STALL_TIMEOUT_SECONDS,
        help="Infrastructure deadman timeout based on worker log/heartbeat activity.",
    )
    return parser


def validate_plan_reuse(previous: dict[str, Any], plan: dict[str, Any]) -> None:
    """Keep a saved trajectory tied to its policy and semantic evaluator."""
    for key in (
        "host_index",
        "host_count",
        "selected_model_bindings",
        "evaluation_support_binding",
    ):
        previous_value = previous.get(key)
        current_value = plan.get(key)
        if (
            key == "evaluation_support_binding"
            and isinstance(previous_value, dict)
            and isinstance(current_value, dict)
        ):
            previous_value = evaluation_support_binding(previous_value)
            current_value = evaluation_support_binding(current_value)
        if previous_value != current_value:
            raise ValueError(
                f"Output directory belongs to a different plan ({key}); "
                "choose a new --output-dir."
            )


def main(args: argparse.Namespace | None = None) -> None:
    global _monitor_cache
    args = args or build_parser().parse_args()
    from tarsphere.harness.harness_cli import agent_harness_config_from_args

    harness_config = agent_harness_config_from_args(args)
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.max_starting_workers < 1:
        raise ValueError("--max-starting-workers must be positive")
    if args.worker_stall_timeout_seconds <= 0:
        raise ValueError("--worker-stall-timeout-seconds must be positive")
    if args.workers_per_gpu and args.reserve_gpu0_for_services:
        raise ValueError("--workers-per-gpu and --reserve-gpu0-for-services conflict")
    if args.launch and (not args.model.strip() or not args.api_base.strip()):
        raise ValueError(
            "Set TARSPHERE_AGENT_MODEL/TARSPHERE_AGENT_API_BASE or pass --model/--api-base"
        )
    episodes = load_episodes(args.episodes, args.episode_id)
    models = [
        ModelEndpoint(
            name="agent",
            model=args.model,
            api_base=args.api_base.rstrip("/"),
            api_key_env=args.api_key_env,
            max_tokens=args.max_tokens,
            api_key_fallback_env=None,
        )
    ]
    jobs = build_jobs(models, episodes)
    from tarsphere.evaluation.judge import JudgeConfig

    judge_config = JudgeConfig.from_environment()

    gpu_count = (
        detected_gpu_count()
        if args.launch
        else (len(args.workers_per_gpu) if args.workers_per_gpu else 1)
    )
    slots = evaluation_slots(
        gpu_count,
        reserve_gpu0_for_services=args.reserve_gpu0_for_services and not args.launch,
        workers_per_gpu=args.workers_per_gpu,
    )
    physical_gpu_indices = list(range(gpu_count))
    if args.launch:
        from tarsphere.cli.gpu_selection import visible_gpu_indices

        physical_gpu_indices = visible_gpu_indices()
        slots = [
            WorkerSlot(physical_gpu_indices[slot.gpu], slot.runtime_slot)
            for slot in slots
        ]
        if args.reserve_gpu0_for_services:
            if 0 not in physical_gpu_indices:
                raise ValueError("Physical GPU 0 is outside CUDA_VISIBLE_DEVICES")
            slots = [slot for slot in slots if slot.gpu != 0]
            if not slots:
                raise ValueError("At least one Isaac worker slot is required")
    cpu_core_count = physical_cpu_cores()
    slots = slots[: len(jobs)]
    cpu_threads_per_worker = max(2, min(8, cpu_core_count // len(slots)))

    host_scene_ids = {scene.job_id for job in jobs for scene in job.scene_runs}
    model_scene_batches = sum(
        len(scene.models) for job in jobs for scene in job.scene_runs
    )
    episode_runs = sum(
        len(scene.models) * len(scene.episodes)
        for job in jobs
        for scene in job.scene_runs
    )
    from tarsphere.source_identity import runtime_source_tree_sha256

    output = args.output_dir.resolve()
    source_files = [
        *(Path(row.path) for row in episodes),
        ROOT / "tarsphere/cli/evaluate.py",
        ROOT / "tarsphere/cli/worker.py",
        ROOT / "tarsphere/evaluation/engine.py",
        ROOT / "tarsphere/evaluation/schema.py",
        ROOT / "tarsphere/evaluation/progress.py",
        ROOT / "tarsphere/evaluation/official.py",
    ]
    from dulwich.repo import Repo
    from dulwich.errors import NotGitRepository

    try:
        with Repo(str(ROOT)) as repository:
            git_head = repository.head().decode("ascii")
    except (NotGitRepository, KeyError):
        # A source distribution need not include version-control metadata;
        # the complete runtime/file hashes below remain authoritative.
        git_head = ""
    plan = {
        "schema_version": "tarsphere.evaluation_plan.v7",
        "agent_harness_config": harness_config.as_dict(),
        "episode_input": str(args.episodes.resolve()),
        "episode_bindings": [
            {"episode_id": row.episode_id, "path": row.path, "sha256": row.sha256}
            for row in episodes
        ],
        "host_index": 0,
        "host_count": 1,
        "worker_topology": {
            "physical_gpu_indices": physical_gpu_indices,
            "workers_per_gpu": [
                sum(slot.gpu == gpu for slot in slots) for gpu in physical_gpu_indices
            ],
            "total_isaac_workers": len(slots),
            "worker_lanes": len(jobs),
            "lanes_with_one_scene": sum(len(job.scene_runs) == 1 for job in jobs),
            "lanes_with_two_scenes": sum(len(job.scene_runs) == 2 for job in jobs),
            "maximum_scene_loads_per_worker": max(len(job.scene_runs) for job in jobs),
            "x_navdp_processes_per_isaac_worker": 1,
            "sam3_instances": 1,
            "graspgenx_instances": 1,
            "physical_cpu_cores": cpu_core_count,
            "math_threads_per_isaac_worker": cpu_threads_per_worker,
        },
        "model_count": len(models),
        "unique_scene_count": len(host_scene_ids),
        "scene_load_count": sum(len(job.scene_runs) for job in jobs),
        "model_scene_run_count": model_scene_batches,
        "episode_run_count": episode_runs,
        "episodes_per_scene_per_model": sorted(
            {len(scene.episodes) for job in jobs for scene in job.scene_runs}
        ),
        "selected_models": [model.name for model in models],
        "selected_model_bindings": [
            {
                "name": model.name,
                "model": model.model,
                "api_base": model.api_base,
                "max_tokens": model.max_tokens,
                "api_key_env": model.api_key_env,
                "api_key_fallback_env": model.api_key_fallback_env,
                "api_key_fallback_2_env": model.api_key_fallback_2_env,
            }
            for model in models
        ],
        "evaluation_support_binding": evaluation_support_binding(judge_config),
        "evaluation_execution": (
            "parallel_online_score_then_offline_failure_attribution"
            if args.evaluate_after_rollout
            else "deferred_offline"
        ),
        "video_recording": "disabled",
        "source_identity": {
            "git_head": git_head,
            "runtime_source_sha256": runtime_source_tree_sha256(ROOT),
            "files": {str(path.resolve()): sha256(path) for path in source_files},
        },
        "slots": [slot.name for slot in slots],
        "slot_assignments": [
            {
                "slot": slot.name,
                "worker_lane_id": job.job_id,
                "scene_load_count": len(job.scene_runs),
                "model_scene_batch_count": model_scene_batch_count(job),
            }
            for slot, job in zip(slots, jobs)
        ],
    }
    if not args.launch:
        print(json.dumps(plan, indent=2))
        return

    for model in models:
        model.api_key()
    if (
        not judge_config.api_key.strip()
        or not judge_config.base_url.strip()
        or not judge_config.model.strip()
    ):
        raise ValueError(
            "Launch requires the top-level "
            "TARSPHERE_EVALUATION_SUPPORT_API_KEY and "
            "TARSPHERE_EVALUATION_SUPPORT_API_BASE configuration."
        )

    output.mkdir(parents=True, exist_ok=True)
    plan_path = control_directory(output) / "host0" / "plan.json"
    if plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_plan_reuse(previous, plan)
    validate_harness_output_config(output, harness_config=harness_config)
    atomic_json(plan_path, plan)

    try:
        check_local_model_services()
    except Exception as error:
        print(
            f"[EVALUATION] shared perception services unavailable: {error}", flush=True
        )
        raise SystemExit(75)
    startup_probe_warnings: list[str] = []
    unavailable_services: set[str] = set()
    for model in models:
        try:
            check_agent_endpoint(model)
        except Exception as exc:  # transient remote outage must not stop the host
            warning = f"{model.name}: {type(exc).__name__}: {exc}"
            startup_probe_warnings.append(warning)
            unavailable_services.add(model.name)
            print(
                f"[EVALUATION] Startup endpoint probe deferred: {warning}",
                flush=True,
            )
    try:
        _check_chat_inference(
            name="evaluation support service",
            model=judge_config.model,
            api_base=judge_config.base_url,
            api_key=judge_config.api_key,
        )
    except Exception as exc:  # saved trajectories permit later judge recovery
        warning = f"evaluation support: {type(exc).__name__}: {exc}"
        startup_probe_warnings.append(warning)
        unavailable_services.add("evaluation_support")
        print(
            f"[EVALUATION] Startup endpoint probe deferred: {warning}",
            flush=True,
        )

    cache_root = Path(
        os.environ.get(
            "TARSPHERE_EVALUATION_CACHE_ROOT",
            f"/tmp/{os.environ.get('USER', 'tarsphere')}_evaluation_cache",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    running: dict[str, RunningWorker] = {}
    last_scheduler_error: str | None = None
    model_by_name = {model.name: model for model in models}
    service_probe_ready_until: dict[str, float] = {}
    service_probe_not_before: dict[str, float] = {}

    def deferred_services_are_ready(job: EvaluationJob) -> bool:
        """Probe recovered endpoints before paying for another Isaac load."""

        pending_models = {
            model.name
            for scene in job.scene_runs
            for model in scene.models
            if pending_episodes(output, scene, model)
        }
        services = job_external_service_outages(output, job) | (
            unavailable_services & (pending_models | {"evaluation_support"})
        )
        unavailable_services.update(services)
        # A missing Agent endpoint must not delay healthy model batches in the
        # same resident scene. The shared support backend is different: it is
        # part of public Human/NPC interaction during physical rollout.
        if (
            "evaluation_support" not in unavailable_services
            and pending_models - unavailable_services
        ):
            return True
        for service in sorted(services):
            now = time.monotonic()
            if service_probe_ready_until.get(service, 0.0) > now:
                # A prior lane may already have proved this shared service
                # healthy.  Persisted retry-exhausted result rows still name
                # it as an outage, so clear the process-wide outage marker
                # before considering this lane blocked.
                unavailable_services.discard(service)
                continue
            not_before = service_probe_not_before.get(service, 0.0)
            if not_before > now:
                continue
            try:
                if service == "evaluation_support":
                    _check_chat_inference(
                        name="evaluation support service",
                        model=judge_config.model,
                        api_base=judge_config.base_url,
                        api_key=judge_config.api_key,
                    )
                else:
                    check_agent_endpoint(model_by_name[service])
            except Exception as exc:
                job.endpoint_deferrals += 1
                delay = retry_delay_seconds(
                    job.job_id,
                    job.endpoint_deferrals,
                    initial_seconds=ENDPOINT_RETRY_INITIAL_SECONDS,
                    cap_seconds=ENDPOINT_RETRY_CAP_SECONDS,
                )
                service_probe_not_before[service] = now + delay
                unavailable_services.add(service)
                job.last_error = (
                    f"{service} recovery probe failed: {type(exc).__name__}: {exc}"
                )
                print(
                    f"[EVALUATION] {job.job_id}: {job.last_error}; "
                    f"next probe in {delay:.1f}s.",
                    flush=True,
                )
                continue
            service_probe_not_before.pop(service, None)
            unavailable_services.discard(service)
            service_probe_ready_until[service] = now + 300.0
            print(
                f"[EVALUATION] {service} recovered; pending lanes may resume.",
                flush=True,
            )
        blocked = unavailable_services & (pending_models | {"evaluation_support"})
        if "evaluation_support" in blocked or not (pending_models - blocked):
            wake_at = min(
                (
                    service_probe_not_before.get(s, time.monotonic() + 60)
                    for s in blocked
                ),
                default=time.monotonic() + 60,
            )
            make_job_retryable(job, delay_seconds=max(1, wake_at - time.monotonic()))
            return False
        return True

    for job in jobs:
        worker_dir = scene_worker_dir(output, job)
        worker_dir.mkdir(parents=True, exist_ok=True)
        existing_sequences = [
            int(path.stem.rsplit("_", 1)[-1])
            for path in worker_dir.glob("launch_*.json")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        ]
        job.launch_sequence = max(existing_sequences, default=0)
        if pending_episode_count(output, job) == 0:
            job.state = "finished"

    retryable_states = {"queued", "cooldown"}
    _monitor_cache = {}
    try:
        while running or any(job.state in retryable_states for job in jobs):
            release_ready_jobs(jobs)
            for slot_name, worker in list(running.items()):
                refresh_worker_activity(worker, output)
                if (
                    worker.process.poll() is None
                    and not worker.stall_terminated
                    and time.monotonic() - worker.last_activity_monotonic
                    > args.worker_stall_timeout_seconds
                ):
                    worker.stall_terminated = True
                    worker.stall_terminated_at = time.monotonic()
                    try:
                        os.killpg(worker.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                elif (
                    worker.process.poll() is None
                    and worker.stall_terminated_at is not None
                    and time.monotonic() - worker.stall_terminated_at > 60.0
                ):
                    try:
                        os.killpg(worker.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                exit_code = worker.process.poll()
                if exit_code is None:
                    continue
                worker.log_file.close()
                try:
                    stop_navdp(worker.navdp_port)
                except Exception as exc:
                    last_scheduler_error = (
                        f"{worker.job.job_id}: X-NavDP cleanup failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"[EVALUATION] {last_scheduler_error}", flush=True)
                worker.job.last_exit_code = exit_code
                physical_pending = pending_episode_count(output, worker.job)
                for service in job_external_service_outages(output, worker.job):
                    unavailable_services.add(service)
                    service_probe_ready_until.pop(service, None)
                try:
                    launch_record = json.loads(
                        worker.launch_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    launch_record = {}
                launch_record.update(
                    {
                        "status": "exited",
                        "exit_code": exit_code,
                        "stall_terminated": worker.stall_terminated,
                        "pending_episode_run_count_after_exit": physical_pending,
                    }
                )
                atomic_json(worker.launch_path, launch_record)
                made_progress = (
                    physical_pending < worker.pending_episode_run_count_at_launch
                )
                if made_progress:
                    worker.job.consecutive_infrastructure_failures = 0
                    worker.job.endpoint_deferrals = 0
                if physical_pending == 0:
                    worker.job.state = "finished"
                    worker.job.last_error = None
                elif exit_code == 0 and (
                    job_has_external_service_outage(output, worker.job)
                    or unavailable_services
                    & {
                        model.name
                        for scene in worker.job.scene_runs
                        for model in scene.models
                    }
                ):
                    worker.job.endpoint_deferrals += 1
                    delay = retry_delay_seconds(
                        worker.job.job_id,
                        worker.job.endpoint_deferrals,
                        initial_seconds=ENDPOINT_RETRY_INITIAL_SECONDS,
                        cap_seconds=ENDPOINT_RETRY_CAP_SECONDS,
                    )
                    worker.job.last_error = (
                        "external model service recovery exhausted; "
                        f"retrying pending episodes after {delay:.1f}s"
                    )
                    make_job_retryable(worker.job, delay_seconds=delay)
                else:
                    if not made_progress:
                        worker.job.consecutive_infrastructure_failures += 1
                    failure_count = worker.job.consecutive_infrastructure_failures
                    worker.job.last_error = (
                        "worker stalled"
                        if worker.stall_terminated
                        else f"worker exited with code {exit_code}"
                    )
                    delay = retry_delay_seconds(
                        worker.job.job_id,
                        max(1, failure_count),
                        initial_seconds=INFRASTRUCTURE_RETRY_INITIAL_SECONDS,
                        cap_seconds=ENDPOINT_RETRY_CAP_SECONDS,
                    )
                    make_job_retryable(worker.job, delay_seconds=delay)
                del running[slot_name]

            release_ready_jobs(jobs)
            starting_count = sum(
                not worker_is_primed(worker) for worker in running.values()
            )
            try:
                memory = free_gpu_memory()
            except Exception as exc:
                memory = {}
                last_scheduler_error = (
                    "GPU-memory probe failed; retaining all workers and "
                    f"retrying without new launches: {type(exc).__name__}: {exc}"
                )
                print(f"[EVALUATION] {last_scheduler_error}", flush=True)
            for slot in slots:
                if starting_count >= args.max_starting_workers:
                    break
                if (
                    slot.name in running
                    or memory.get(slot.gpu, 0) < args.min_free_memory_mib
                ):
                    continue
                if slot.runtime_slot > 0:
                    preceding = running.get(
                        f"gpu{slot.gpu}_slot{slot.runtime_slot - 1}"
                    )
                    if preceding is not None and not worker_is_primed(preceding):
                        continue
                job = None
                for candidate in jobs:
                    if candidate.state != "queued":
                        continue
                    if deferred_services_are_ready(candidate):
                        job = candidate
                        break
                if job is None:
                    break

                port = navdp_port(slot, args.navdp_port_base)
                job.launch_sequence += 1
                launch_sequence = job.launch_sequence
                worker_dir = scene_worker_dir(output, job)
                schedule_path = worker_dir / f"schedule_{launch_sequence}.json"
                atomic_json(
                    schedule_path,
                    schedule_document(
                        output, job, excluded_models=unavailable_services
                    ),
                )
                command = worker_command(
                    schedule_path,
                    evaluate_after_rollout=args.evaluate_after_rollout,
                    harness_config=harness_config,
                )
                log_path = worker_dir / f"worker_launch_{launch_sequence}.log"
                launch_path = worker_dir / f"launch_{launch_sequence}.json"
                pending_at_launch = pending_episode_count(output, job)
                launch_record = {
                    "command": command,
                    "slot": slot.name,
                    "navdp_port": port,
                    "pending_episode_run_count": pending_at_launch,
                    "status": "starting",
                }
                atomic_json(launch_path, launch_record)
                log_file = None
                try:
                    start_navdp(slot, port)
                    log_file = log_path.open("x", encoding="utf-8")
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=worker_environment(
                            slot, port, cache_root, cpu_threads_per_worker
                        ),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except Exception as exc:
                    if log_file is not None:
                        log_file.close()
                    cleanup_error = None
                    try:
                        stop_navdp(port)
                    except Exception as cleanup:
                        cleanup_error = f"{type(cleanup).__name__}: {cleanup}"
                    job.consecutive_infrastructure_failures += 1
                    job.last_error = (
                        f"worker launch failed: {type(exc).__name__}: {exc}"
                    )
                    if cleanup_error:
                        job.last_error += f"; X-NavDP cleanup failed: {cleanup_error}"
                    launch_record.update(
                        {
                            "status": "launch_failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "cleanup_error": cleanup_error,
                        }
                    )
                    atomic_json(launch_path, launch_record)
                    print(
                        f"[EVALUATION] {job.job_id}: {job.last_error}",
                        flush=True,
                    )
                    delay = retry_delay_seconds(
                        job.job_id,
                        job.consecutive_infrastructure_failures,
                        initial_seconds=INFRASTRUCTURE_RETRY_INITIAL_SECONDS,
                        cap_seconds=ENDPOINT_RETRY_CAP_SECONDS,
                    )
                    make_job_retryable(job, delay_seconds=delay)
                    continue
                launch_record.update({"status": "running", "pid": process.pid})
                atomic_json(launch_path, launch_record)
                job.state = "running"
                job.last_error = None
                running[slot.name] = RunningWorker(
                    process=process,
                    job=job,
                    slot=slot,
                    log_path=log_path,
                    launch_path=launch_path,
                    log_file=log_file,
                    navdp_port=port,
                    last_activity_monotonic=time.monotonic(),
                    pending_episode_run_count_at_launch=pending_at_launch,
                )
                starting_count += 1
                memory[slot.gpu] = 0
                time.sleep(8)

            atomic_json(
                plan_path.parent / "status.json",
                {
                    "updated_at_unix": time.time(),
                    "startup_probe_warnings": startup_probe_warnings,
                    "last_scheduler_error": last_scheduler_error,
                    "counts": dict(Counter(job.state for job in jobs)),
                    "completed_physical_episode_run_count": sum(
                        sum(
                            len(scene.models) * len(scene.episodes)
                            for scene in job.scene_runs
                        )
                        - pending_episode_count(output, job)
                        for job in jobs
                    ),
                    "evaluated_episode_run_count": sum(
                        sum(
                            len(scene.models) * len(scene.episodes)
                            for scene in job.scene_runs
                        )
                        - unfinished_episode_count(output, job)
                        for job in jobs
                    ),
                    "total_episode_run_count": plan["episode_run_count"],
                    "running": {
                        slot_name: {
                            "pid": worker.process.pid,
                            "worker_lane_id": worker.job.job_id,
                            "navdp_port": worker.navdp_port,
                            "primed": worker_is_primed(worker),
                            "stall_terminated": worker.stall_terminated,
                            "seconds_since_activity": round(
                                time.monotonic() - worker.last_activity_monotonic, 1
                            ),
                        }
                        for slot_name, worker in running.items()
                    },
                    "jobs": [job_record(job, output) for job in jobs],
                },
            )
            if running or any(job.state in retryable_states for job in jobs):
                time.sleep(15)
    finally:
        _monitor_cache = None
        for worker in running.values():
            if worker.process.poll() is None:
                os.killpg(worker.process.pid, signal.SIGTERM)
                try:
                    worker.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.process.pid, signal.SIGKILL)
                    worker.process.wait(timeout=10)
            worker.log_file.close()
            try:
                stop_navdp(worker.navdp_port)
            except Exception as exc:
                print(
                    "[EVALUATION] X-NavDP cleanup failed during shutdown: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    # Monitoring caches cannot authorize successful reuse or completion.
    failures = [
        job.job_id
        for job in jobs
        if job.state != "finished" or pending_episode_count(output, job)
    ]
    if failures:
        print(f"{len(failures)} scene jobs did not finish: {failures[:10]}", flush=True)
        raise SystemExit(75)
    pending_evaluations = sum(evaluation_pending_count(output, job) for job in jobs)
    print(
        f"[EVALUATION] Finished {plan['episode_run_count']} physical episode "
        f"runs; {pending_evaluations} saved trajectories await offline evaluation."
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        with ExitStack() as ownership:
            if args.launch:
                require_parent_lifetime("TARSPHERE_SUPERVISOR_PID")
                ownership.enter_context(
                    exclusive_lock(host_state_directory() / "physical.lock")
                )
                ownership.enter_context(
                    exclusive_lock(
                        control_directory(args.output_dir.resolve())
                        / "host0"
                        / "results.lock"
                    )
                )

                def interrupt(_signum, _frame):
                    raise KeyboardInterrupt

                signal.signal(signal.SIGTERM, interrupt)
            main(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, FileExistsError, RuntimeError) as error:
        print(f"[EVALUATION] blocked: {error}", file=sys.stderr)
        raise SystemExit(78)
