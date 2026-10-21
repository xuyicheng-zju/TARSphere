"""Clean presentation cameras above one causal, shared evidence panel.

Presentation video samples use the simulator's 240 Hz clock. Each input frame
manifest distinguishes the captured source tick from its uniform video tick;
alignment always selects a frame from the past, never a future observation.
This module only reads recordings and cannot alter policy observations.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

from tarsphere.infra.control.physical_contracts import SIMULATION_PHYSICS_HZ
from tarsphere.infra.perception.model_output_overlay import (
    _FocusAss,
    _action_caption,
    _current_task_text,
    _overlay_decision,
    _overlay_utterances,
    _plan_states_by_trace_index,
    decision_indices_by_physics_step,
    ffmpeg_subtitles_filter,
    write_online_map_video,
)
from tarsphere.infra.perception.presentation_radar import execution_feedback_caption


PHYSICS_HZ = int(SIMULATION_PHYSICS_HZ)
VIDEO_FPS = 30
VIDEO_STRIDE = PHYSICS_HZ // VIDEO_FPS


def file_sha256(path):
    """Digest recorded artifacts without loading full-resolution videos in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelObservation:
    """A hash-verified provider ego image and its measured camera capture tick."""

    physics_step: int
    capture_id: int
    policy_step: int
    action_step: int
    image_path: Path
    image_sha256: str
    request_path: Path
    snapshot_sha256: str
    request_sha256: str

    def provenance(self):
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self).items()
        }


@dataclass(frozen=True)
class ConditionedDepthObservation:
    """Actual runtime depth visualization; never a provider image or metric array."""

    physics_step: int
    capture_id: int
    action_step: int
    policy_inference: bool
    observation_id: str
    depth_view_path: Path
    depth_view_sha256: str
    source_rgbd_path: Path
    source_rgbd_sha256: str
    depth_resolution: list
    crop_box: list
    kind: str
    scope: str

    @property
    def image_path(self):
        return self.depth_view_path

    def provenance(self):
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self).items()
        }


def _recorded_artifact_path(value, *, result_dir, source_result_root=None):
    """Map one original absolute artifact path into an explicitly copied take."""
    local_root = Path(result_dir).resolve()
    recorded = Path(value)
    if source_result_root is None:
        path = recorded.resolve()
    else:
        source_root = Path(source_result_root)
        if (
            not source_root.is_absolute()
            or not recorded.is_absolute()
            or ".." in source_root.parts
            or ".." in recorded.parts
        ):
            raise ValueError(
                "Source result mapping requires absolute paths without traversal"
            )
        try:
            relative = recorded.relative_to(source_root)
        except ValueError:
            raise ValueError(
                "Recorded artifact is outside the explicit source result root"
            ) from None
        path = (local_root / relative).resolve()
    if not path.is_relative_to(local_root):
        raise ValueError("Recorded artifact must belong to this recorded take")
    return path


def load_recorded_rollout(*, result, result_dir, source_result_root=None):
    """Read the result's hash-bound physical trace, never a neighboring take."""
    result_dir = Path(result_dir).resolve()
    rollout_path = _recorded_artifact_path(
        result["rollout_artifact_path"],
        result_dir=result_dir,
        source_result_root=source_result_root,
    )
    raw = rollout_path.read_bytes()
    rollout_digest = hashlib.sha256(raw).hexdigest()
    if rollout_digest != result["rollout_artifact_sha256"]:
        raise ValueError("Recorded rollout hash mismatch")
    rollout = json.loads(raw)
    if rollout.get("schema_version") != "tarsphere.rollout_artifact.v1":
        raise ValueError("Demo requires the immutable rollout schema")
    if rollout["episode_id"] != result["episode_id"] or (
        "trace" in result and rollout["trace"] != result["trace"]
    ):
        raise ValueError("Recorded rollout belongs to a different episode/trace")
    receipt = {"path": str(rollout_path), "sha256": rollout_digest}
    if source_result_root is not None:
        receipt["recorded_path"] = result["rollout_artifact_path"]
    return rollout, receipt


def load_editorial_annotations(
    *,
    path,
    result_path,
    result,
    rollout,
    heartbeats,
    heartbeats_path,
    source_result_root=None,
):
    """Validate a separate author's review against sealed source and provider text.

    Intervals are inclusive source ticks within the original policy's display
    interval. They never change the thinking, outcome, action or frame clocks.
    """
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "tarsphere.demo.editorial_annotations.v1":
        raise ValueError("Unsupported editorial annotation schema")
    required = (
        "episode_id",
        "result_sha256",
        "rollout_sha256",
        "episode_sha256",
        "source_episode_path",
        "source_episode_sha256",
        "annotations",
    )
    if any(key not in document for key in required):
        raise ValueError("Editorial annotations require every sealed source identity")
    if (
        document["episode_id"] != result["episode_id"]
        or rollout["episode_id"] != result["episode_id"]
        or document["result_sha256"] != file_sha256(result_path)
        or document["rollout_sha256"] != result["rollout_artifact_sha256"]
        or document["episode_sha256"] != result["episode_artifact_sha256"]
        or rollout["episode_artifact_sha256"] != result["episode_artifact_sha256"]
    ):
        raise ValueError(
            "Editorial annotation episode/result/rollout identity mismatch"
        )
    # Re-read the hash-bound rollout rather than trusting an unrelated dict.
    sealed, _ = load_recorded_rollout(
        result=result,
        result_dir=Path(result_path).parent,
        source_result_root=source_result_root,
    )
    if (
        sealed != rollout
        or heartbeats != rollout["heartbeats"]
        or file_sha256(heartbeats_path) != result["heartbeat_sha256"]
        or rollout["heartbeat_sha256"] != result["heartbeat_sha256"]
    ):
        raise ValueError("Editorial annotation rollout/heartbeat clock mismatch")
    episode_path = _recorded_artifact_path(
        result["episode_artifact_path"],
        result_dir=Path(result_path).parent,
        source_result_root=source_result_root,
    )
    if (
        rollout["episode_artifact_path"] != result["episode_artifact_path"]
        or file_sha256(episode_path) != document["episode_sha256"]
    ):
        raise ValueError("Editorial annotation sealed episode hash mismatch")
    source_path = Path(document["source_episode_path"])
    if not source_path.is_absolute():
        source_path = Path(__file__).resolve().parents[3] / source_path
    if file_sha256(source_path) != document["source_episode_sha256"]:
        raise ValueError("Editorial annotation original source SHA mismatch")
    from tarsphere.bench.case_loader import BenchmarkCaseLoader

    if BenchmarkCaseLoader().load_file(source_path).model_dump(
        mode="json"
    ) != json.loads(episode_path.read_bytes()):
        raise ValueError(
            "Editorial annotation original source differs from sealed episode"
        )
    trace = rollout["trace"]
    if len(heartbeats) != len(trace):
        raise ValueError("Editorial annotations require the complete action clock")
    endpoints = [row.get("physics_steps") for row in heartbeats]
    if any(type(tick) is not int or tick < 0 for tick in endpoints) or any(
        b < a for a, b in zip(endpoints, endpoints[1:])
    ):
        raise ValueError("Editorial annotation action clock is not chronological")
    rows = document["annotations"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Editorial annotations require a nonempty author annotation list"
        )
    validated, previous_end = [], -1
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Editorial annotation must be an object")
        for key in (
            "action_step",
            "policy_step",
            "source_physics_start",
            "source_physics_end",
        ):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(
                    "Editorial annotation indices/ticks must be nonnegative integers"
                )
        matches = [
            i
            for i, record in enumerate(trace)
            if record.get("action_step") == row["action_step"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "Editorial annotation action does not identify one recorded decision"
            )
        index = matches[0]
        record = trace[index]
        quote, note = row.get("thinking_quote"), row.get("note")
        if (
            record.get("policy_step") != row["policy_step"]
            or record.get("policy_inference") is not True
        ):
            raise ValueError(
                "Editorial annotation policy must identify the original inference"
            )
        if (
            not isinstance(quote, str)
            or not quote.strip()
            or not isinstance(note, str)
            or not note.strip()
            or row.get("kind") not in ("review", "highlight")
        ):
            raise ValueError(
                "Editorial annotation requires a quote, author note and kind"
            )
        thinking = record.get("envelope", {}).get("thinking")
        if not isinstance(thinking, str) or thinking.count(quote) != 1:
            raise ValueError(
                "Editorial annotation quote is absent from recorded thinking"
            )
        calls = record.get("model_calls") or []
        provider_thinking = []
        for call in calls:
            for response in call.get("responses", []):
                try:
                    content = json.loads(response.get("content", ""))
                except (ValueError, TypeError):
                    continue
                if isinstance(content, dict):
                    provider_thinking.append(content.get("thinking"))
        if len(calls) != 1 or thinking not in provider_thinking:
            raise ValueError(
                "Editorial annotation thinking lacks its exact recorded provider response"
            )
        policy_indices = [index]
        for following in range(index + 1, len(trace)):
            queued = trace[following]
            if queued.get("policy_step") != row["policy_step"]:
                break
            queued_envelope = queued.get("envelope", {})
            action_index = queued.get("action_index")
            original_actions = record["envelope"]["actions"]
            if (
                queued.get("policy_inference") is not False
                or queued.get("model_calls")
                or queued_envelope.get("thinking", thinking) != thinking
                or type(action_index) is not int
                or not 0 <= action_index < len(original_actions)
                or queued.get("action_sequence_length") != len(original_actions)
                or queued_envelope.get("actions") != [original_actions[action_index]]
            ):
                raise ValueError(
                    "Editorial annotation queued policy thinking/identity mismatch"
                )
            policy_indices.append(following)
        begin, finish = row["source_physics_start"], row["source_physics_end"]
        earliest = endpoints[index - 1] + 1 if index else 0
        if not (
            previous_end < begin <= finish <= endpoints[policy_indices[-1]]
            and begin >= earliest
        ):
            raise ValueError(
                "Editorial annotation ticks overlap, are out of order or leave the original policy"
            )
        previous_end = finish
        validated.append(
            {
                **row,
                "trace_index": index,
                "policy_trace_indices": policy_indices,
                "thinking_sha256": hashlib.sha256(thinking.encode("utf-8")).hexdigest(),
                "thinking_display_offset": len(thinking) - len(thinking.lstrip()),
            }
        )
    return validated, {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema": document["schema"],
        "source_episode_path": str(source_path.resolve()),
        "source_episode_sha256": document["source_episode_sha256"],
        "episode_sha256": document["episode_sha256"],
        "result_sha256": document["result_sha256"],
        "rollout_sha256": document["rollout_sha256"],
        "annotations": rows,
        "scope": "Author editorial review; original model thinking and source clock unchanged",
    }


def annotated_panel_states(
    physics_steps, trace, heartbeats, human_replay_log, annotations
):
    states = panel_states(physics_steps, trace, heartbeats, human_replay_log)
    indices = decision_indices_by_physics_step(physics_steps, trace, heartbeats)
    for tick, index, state in zip(physics_steps, indices, states):
        state["editorial_annotation"] = next(
            (
                row
                for row in annotations
                if index in row["policy_trace_indices"]
                and row["source_physics_start"] <= tick <= row["source_physics_end"]
            ),
            None,
        )
        if state["editorial_annotation"]:
            state["editorial_display_action_step"] = trace[index]["action_step"]
    return states


def recorded_sensor_panel(*, result, result_dir, source_result_root=None):
    """Select evidence from the hash-bound recorded episode, never its filename."""
    rollout, _ = load_recorded_rollout(
        result=result, result_dir=result_dir, source_result_root=source_result_root
    )
    path = _recorded_artifact_path(
        result["episode_artifact_path"],
        result_dir=result_dir,
        source_result_root=source_result_root,
    )
    expected = result["episode_artifact_sha256"]
    if (
        not path.is_relative_to(Path(result_dir).resolve())
        or rollout["episode_artifact_path"] != result["episode_artifact_path"]
        or rollout["episode_artifact_sha256"] != expected
        or file_sha256(path) != expected
    ):
        raise ValueError(
            "Presentation sensor condition requires its hash-bound episode artifact"
        )
    episode = json.loads(path.read_bytes())
    if episode["episode_id"] != result["episode_id"]:
        raise ValueError("Presentation sensor condition belongs to another episode")
    profile = episode["evaluation"]["task_profile"]
    if profile["capability_slice"] != "perceptual_robustness":
        return "observed_spatial_memory"
    if profile["subtype"] == "depth_noise":
        return "grounding_depth"
    if profile["subtype"] in {"low_light", "reduced_resolution"}:
        return "model_observation"
    raise ValueError("Unknown recorded perceptual sensor condition")


def load_conditioned_depth_observations(*, result, result_dir, source_result_root=None):
    """Verify exact pre-agent clocks, original RGBD bytes and right-half pixels.

    The live source images may have been cleaned up by the worker. Its retained
    byte copy and crop must live in this exact episode/attempt's sensor directory
    and be identical to the records sealed into the immutable physical rollout.
    Queued actions retain their own observation identity, without inventing a
    policy request for them.
    """
    from PIL import Image

    root = Path(result_dir).resolve()
    rollout, provenance = load_recorded_rollout(
        result=result, result_dir=root, source_result_root=source_result_root
    )
    if (
        recorded_sensor_panel(
            result=result, result_dir=root, source_result_root=source_result_root
        )
        != "grounding_depth"
    ):
        raise ValueError(
            "Grounding depth panel requires the recorded depth_noise condition"
        )
    captures = result.get("conditioned_depth_captures")
    if not captures or captures != rollout.get("conditioned_depth_captures"):
        raise ValueError(
            "Conditioned depth evidence must equal the immutable rollout records"
        )
    manifest = _recorded_artifact_path(
        result["continuous_ego_videos"]["frame_manifest"],
        result_dir=root,
        source_result_root=source_result_root,
    )
    suffix = "_ego_frames.jsonl"
    if not manifest.is_relative_to(root) or not manifest.name.endswith(suffix):
        raise ValueError("Conditioned depth requires this recorded attempt identity")
    slug = manifest.name.removesuffix(suffix)
    capture_root = root / ".observer_frames" / slug
    evidence_root = (root / "sensor_evidence" / slug).resolve()
    steps = {step["decision_step"]: step for step in rollout["evaluation_steps"]}
    trace = {record["action_step"]: record for record in rollout["trace"]}
    indices = [record["action_step"] for record in captures]
    if (
        len(steps) != len(rollout["evaluation_steps"])
        or len(trace) != len(rollout["trace"])
        or indices != list(steps)
        or indices != list(trace)
    ):
        raise ValueError(
            "Conditioned depth requires exactly one capture binding for every action, including queued actions"
        )
    observations, checked_pixels = [], set()
    for record in captures:
        index = record["action_step"]
        step, action = steps[index], trace[index]
        observation = step["observation"]
        clock = observation.get("camera_capture") or {}
        if (
            any(
                type(record.get(key)) is not int or record[key] < 0
                for key in ("physics_step", "capture_id", "action_step")
            )
            or any(
                type(clock.get(key)) is not int or clock[key] < 0
                for key in ("physics_step", "capture_id")
            )
            or type(record.get("policy_inference")) is not bool
            or record["policy_inference"] != step["policy_inference"]
            or record["policy_inference"] != action["policy_inference"]
            or observation["timing"] != "pre_agent"
            or observation["step"] != index
            or observation["id"] != record["observation_id"]
            or observation["id"] != action["images"]["pre"]
            or any(record[key] != clock[key] for key in ("physics_step", "capture_id"))
        ):
            raise ValueError(
                "Conditioned depth does not match its real pre-agent action/capture clock"
            )
        if (
            record["kind"] != "runtime_conditioned_depth_visualization"
            or record["depth_resolution"] != [640, 480]
            or record["crop_box"] != [640, 0, 1280, 480]
        ):
            raise ValueError(
                "Conditioned depth requires the native RGBD right-half crop"
            )
        tick, capture = record["physics_step"], record["capture_id"]
        if observations and tick < observations[-1].physics_step:
            raise ValueError("Conditioned depth capture clocks are not chronological")
        source = _recorded_artifact_path(
            record["source_rgbd_path"],
            result_dir=root,
            source_result_root=source_result_root,
        )
        depth = _recorded_artifact_path(
            record["depth_view_path"],
            result_dir=root,
            source_result_root=source_result_root,
        )
        name = f"step_{tick:06d}_capture_{capture:06d}"
        original = _recorded_artifact_path(
            observation["image_path"],
            result_dir=root,
            source_result_root=source_result_root,
        )
        if source_result_root is not None:
            for call in action.get("model_calls", []):
                _recorded_artifact_path(
                    call["input_snapshot"]["path"],
                    result_dir=root,
                    source_result_root=source_result_root,
                )
        if (
            source != evidence_root / (name + ".rgbd.png")
            or depth != evidence_root / (name + ".depth.png")
            or not source.is_relative_to(root)
            or not depth.is_relative_to(root)
            or original != capture_root / (name + "_rgbd.png")
        ):
            raise ValueError(
                "Conditioned depth source escapes the recorded episode/attempt capture directory"
            )
        identity = (
            tick,
            capture,
            record["source_rgbd_sha256"],
            record["depth_view_sha256"],
        )
        if identity not in checked_pixels:
            if (
                file_sha256(source) != record["source_rgbd_sha256"]
                or file_sha256(depth) != record["depth_view_sha256"]
            ):
                raise ValueError("Conditioned depth image hash mismatch")
            with Image.open(source) as rgbd, Image.open(depth) as image:
                if (
                    rgbd.format != "PNG"
                    or image.format != "PNG"
                    or rgbd.mode != "RGB"
                    or image.mode != "RGB"
                ):
                    raise ValueError(
                        "Conditioned depth evidence must retain native RGB PNGs"
                    )
                if rgbd.size != (1280, 480) or image.size != (640, 480):
                    raise ValueError(
                        "Conditioned depth image dimensions do not match the native sensor"
                    )
                if rgbd.crop((640, 0, 1280, 480)).tobytes() != image.tobytes():
                    raise ValueError(
                        "Conditioned depth pixels differ from the actual RGBD right half"
                    )
            checked_pixels.add(identity)
        observations.append(
            ConditionedDepthObservation(
                **{**record, "depth_view_path": depth, "source_rgbd_path": source}
            )
        )
    return observations, provenance


def load_model_observations(
    *, result, result_dir, model_input_dir, source_result_root=None
):
    """Bind exact provider images to the immutable rollout's camera clock.

    The request itself stores policy/action indices, not simulation timestamps.
    Its matching pre-agent camera_capture supplies the measured tick. A missing
    clock is an error; an action-end heartbeat is not substituted for it.
    """
    from PIL import Image
    from tarsphere.harness.kernel.model_input_snapshot import load_model_input_snapshot

    root = Path(model_input_dir).resolve()
    if not root.is_dir() or not root.is_relative_to(Path(result_dir).resolve()):
        raise ValueError("Model-input directory must belong to this recorded take")
    rollout, provenance = load_recorded_rollout(
        result=result, result_dir=result_dir, source_result_root=source_result_root
    )
    steps = {step["decision_step"]: step for step in rollout["evaluation_steps"]}
    if len(steps) != len(rollout["evaluation_steps"]):
        raise ValueError("Duplicate action indices in model-observation clock")
    if source_result_root is not None:
        manifest = _recorded_artifact_path(
            result["continuous_ego_videos"]["frame_manifest"],
            result_dir=result_dir,
            source_result_root=source_result_root,
        )
        suffix = "_ego_frames.jsonl"
        if not manifest.name.endswith(suffix):
            raise ValueError(
                "Relocated model observations require this recorded attempt identity"
            )
        capture_root = (
            Path(result_dir).resolve()
            / ".observer_frames"
            / manifest.name.removesuffix(suffix)
        )
        actions = {record["action_step"]: record for record in rollout["trace"]}
        if len(actions) != len(rollout["trace"]) or actions.keys() != steps.keys():
            raise ValueError(
                "Relocated model observations require every recorded action capture"
            )
        missing_paths = any(
            "image_path" not in step["observation"] for step in steps.values()
        )
        if missing_paths:
            # Initial observation logging may omit image_path. Its immutable
            # capture identity remains independent of the action-end heartbeat.
            attempt = rollout.get("physical_attempt")
            if type(attempt) is not int or attempt < 1:
                raise ValueError(
                    "Pathless model capture requires its recorded physical attempt"
                )
            slug = f"{rollout['episode_id']}_physical_attempt_{attempt:03d}"
            if (
                manifest.name != slug + suffix
                or root.name != slug + ".model_output.inputs"
            ):
                raise ValueError(
                    "Pathless model capture differs from the recorded episode/attempt"
                )
            manifest_bytes = manifest.read_bytes()
            if (
                hashlib.sha256(manifest_bytes).hexdigest()
                != result["continuous_ego_video_sha256"]["frame_manifest"]
            ):
                raise ValueError("Pathless model capture frame manifest hash mismatch")
            frame_ticks = [
                json.loads(line)["physics_step"]
                for line in manifest_bytes.splitlines()
                if line.strip()
            ]
            if (
                not frame_ticks
                or any(type(tick) is not int or tick < 0 for tick in frame_ticks)
                or any(b < a for a, b in zip(frame_ticks, frame_ticks[1:]))
            ):
                raise ValueError(
                    "Pathless model capture requires the recorded source frame clock"
                )
            provenance["capture_identity_without_image_path"] = []
        previous_tick = -1
        for index, step in steps.items():
            observation = step["observation"]
            clock = observation.get("camera_capture") or {}
            if (
                any(
                    type(clock.get(key)) is not int or clock[key] < 0
                    for key in ("physics_step", "capture_id")
                )
                or observation["timing"] != "pre_agent"
                or observation["step"] != index
                or observation["id"] != actions[index]["images"]["pre"]
                or clock["physics_step"] < previous_tick
            ):
                raise ValueError(
                    "Relocated model observations require their actual pre-agent capture identities"
                )
            if "image_path" in observation:
                original = _recorded_artifact_path(
                    observation["image_path"],
                    result_dir=result_dir,
                    source_result_root=source_result_root,
                )
                name = f"step_{clock['physics_step']:06d}_capture_{clock['capture_id']:06d}_rgbd.png"
                if original != capture_root / name:
                    raise ValueError(
                        "Relocated model capture path differs from its recorded episode/attempt clock"
                    )
            else:
                if (
                    type(index) is not int
                    or index < 0
                    or observation["id"] != f"step_{index:03d}_pre_agent"
                    or not frame_ticks[0] <= clock["physics_step"] <= frame_ticks[-1]
                    or type(step.get("physics_step")) is not int
                    or clock["physics_step"] > step["physics_step"]
                ):
                    raise ValueError(
                        "Pathless model capture lacks a verified pre-agent source identity"
                    )
                provenance["capture_identity_without_image_path"].append(
                    {"action_step": index, "observation_id": observation["id"], **clock}
                )
            previous_tick = clock["physics_step"]
    observations = []
    for record in rollout["trace"]:
        if not record["policy_inference"]:
            continue
        calls = record.get("model_calls") or []
        if len(calls) != 1 or not calls[0].get("input_snapshot"):
            raise ValueError(
                "Every displayed policy inference requires its real model-input snapshot"
            )
        reference = dict(calls[0]["input_snapshot"])
        name = Path(reference["path"]).name
        if not re.fullmatch(r"\d{4,}\.request\.json", name):
            raise ValueError("Unexpected model-input snapshot filename")
        path = root / name
        if source_result_root is not None:
            mapped = _recorded_artifact_path(
                reference["path"],
                result_dir=result_dir,
                source_result_root=source_result_root,
            )
            if mapped != path.resolve():
                raise ValueError(
                    "Model-input directory differs from the recorded source episode/attempt"
                )
        # An explicitly supplied local take may relocate a hash-bound archive.
        # The official loader verifies every image and reconstructed request.
        reference["path"] = str(path)
        restored = load_model_input_snapshot(reference, artifact_root=root)
        expected = {
            "policy_step": record["policy_step"],
            "action_step": record["action_step"],
        }
        if restored["context"] != expected:
            raise ValueError(
                "Model-input policy/action indices do not match the displayed trace"
            )
        if restored["request"]["model"] != calls[0]["model"]:
            raise ValueError(
                "Model-input snapshot belongs to a different evaluated model"
            )
        snapshot = json.loads(path.read_bytes())
        if source_result_root is not None:
            for item in snapshot["images"].values():
                relative = Path(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        "Relocated provider images require snapshot-relative paths without traversal"
                    )
        matches = []
        for message in snapshot["request"]["messages"]:
            if message.get("role") != "user" or not isinstance(
                message.get("content"), list
            ):
                continue
            blocks = message["content"]
            for index, block in enumerate(blocks):
                marker = re.fullmatch(
                    r"### Current — ego RGB — step (\d+); normalized image coordinates refer to this complete image",
                    str(block.get("text") or ""),
                )
                if not marker:
                    continue
                if int(marker.group(1)) != record["action_step"]:
                    raise ValueError(
                        "Model-input ego image label has a different action step"
                    )
                if (
                    index + 1 == len(blocks)
                    or blocks[index + 1].get("type") != "image_url"
                ):
                    raise ValueError(
                        "Current ego image marker is not followed by its image"
                    )
                occurrence = blocks[index + 1]["image_url"]["url"]
                matches.append(snapshot["images"][occurrence["image_ref"]])
        if len(matches) != 1:
            raise ValueError(
                "Model observation requires exactly one current ego RGB request image"
            )
        image_info = matches[0]
        image_path = (root / image_info["path"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Model observation image escapes its recorded take")
        with Image.open(image_path) as image:
            if image.size != (640, 480):
                raise ValueError(
                    f"Model observation must be the real 640 × 480 input, got {image.size}"
                )
        step = steps[record["action_step"]]
        observation = step["observation"]
        if (
            not step["policy_inference"]
            or observation["step"] != record["action_step"]
            or observation["timing"] != "pre_agent"
            or observation["id"] != record["images"]["pre"]
        ):
            raise ValueError(
                "Model observation does not match its pre-agent camera capture"
            )
        clock = observation.get("camera_capture") or {}
        if any(
            type(clock.get(key)) is not int or clock[key] < 0
            for key in ("physics_step", "capture_id")
        ):
            raise ValueError(
                "Model observation requires measured camera_capture.physics_step and capture_id"
            )
        if observations and clock["physics_step"] < observations[-1].physics_step:
            raise ValueError("Model-observation capture ticks are not chronological")
        observations.append(
            ModelObservation(
                clock["physics_step"],
                clock["capture_id"],
                record["policy_step"],
                record["action_step"],
                image_path,
                image_info["sha256"],
                path,
                reference["sha256"],
                reference["request_sha256"],
            )
        )
    if not observations:
        raise ValueError(
            "No real policy observation images were recorded for this take"
        )
    return observations, provenance


def sensor_observation_indices(physics_steps, observations):
    """Hold the last measured sensor image; never pull a future capture backwards."""
    ticks = [observation.physics_step for observation in observations]
    return [bisect_right(ticks, tick) - 1 for tick in physics_steps]


def write_sensor_observation_video(
    *, output_path, physics_steps, observations, resolution, image_rect=None
):
    """Hold actual sensor pixels; an explicit image area contains the complete view.

    Full HUD reserves header/footer space in this area. Default dual/below
    cards keep their existing left image and right caption region.
    """
    from PIL import Image
    import numpy as np
    from tarsphere.infra.perception.continuous_ego_video import _RawRgbVideoWriter

    width, height = resolution
    image_width = round(height * 4 / 3)
    if image_rect is not None:
        x, y, area_width, area_height = image_rect
        if (
            any(type(value) is not int for value in image_rect)
            or x < 0
            or y < 0
            or area_width <= 0
            or area_height <= 0
            or x + area_width > width
            or y + area_height > height
        ):
            raise ValueError("Sensor image area must lie inside the evidence card")
    writer = _RawRgbVideoWriter(
        output_path=output_path, resolution=resolution, fps=VIDEO_FPS
    )
    cached_index, pixels = None, None
    try:
        for index in sensor_observation_indices(physics_steps, observations):
            if index != cached_index:
                canvas = Image.new(
                    "RGB",
                    resolution,
                    "#F9F7FB" if image_rect is not None else "#15171E",
                )
                if index >= 0:
                    with Image.open(observations[index].image_path) as image:
                        if image_rect is None:
                            size, position = (image_width, height), (0, 0)
                        else:
                            if image.size != (640, 480):
                                raise ValueError(
                                    "Full HUD sensor source must be the actual 640 × 480 view"
                                )
                            scale = min(
                                area_width / image.width, area_height / image.height
                            )
                            size = (
                                round(image.width * scale),
                                round(image.height * scale),
                            )
                            position = (
                                x + (area_width - size[0]) // 2,
                                y + (area_height - size[1]) // 2,
                            )
                        canvas.paste(
                            image.convert("RGB").resize(size, Image.Resampling.LANCZOS),
                            position,
                        )
                pixels = np.asarray(canvas)
                cached_index = index
            writer.write(pixels)
    finally:
        writer.close()
    return Path(output_path)


@dataclass(frozen=True)
class DemoLayout:
    """16:9 output; aspect-preserving cameras share a common height."""

    height: int = 1440

    def __post_init__(self):
        if self.height not in (720, 1080, 1440, 2160):
            raise ValueError("Demo height must be 720, 1080, 1440, or 2160")

    @property
    def resolution(self):
        return self.height * 16 // 9, self.height

    def rect(self, name):
        # Coordinates are authored on a 1280 x 720 canvas. Cameras contain no UI.
        rect = {
            "ego": (24, 64, 520, 390),
            "observer": (562, 64, 694, 390),
            "map": (866, 505, 390, 170),
        }[name]
        return tuple(round(v * self.height / 720 / 2) * 2 for v in rect)


@dataclass(frozen=True)
class SingleEgoLayout:
    """Native ego pixels at the top; the existing shared panel sits below."""

    width: int
    ego_height: int

    @property
    def resolution(self):
        return self.width, self.ego_height + round(258 * self.width / 1280 / 2) * 2

    def rect(self, name):
        if name == "ego":
            return 0, 0, self.width, self.ego_height
        if name != "map":
            raise KeyError(name)
        q = self.width / 1280
        return tuple(
            round(v / 2) * 2
            for v in (866 * q, self.ego_height + 43 * q, 390 * q, 170 * q)
        )


class _BelowEgoAss(_FocusAss):
    """Reuse panel typography with its authored coordinates shifted below ego."""

    def __init__(self, layout):
        super().__init__(*layout.resolution)
        self.q = layout.width / 1280
        self.panel_offset = layout.ego_height / self.q - 462

    def position(self, x, y, animate=False):
        return super().position(x, y + self.panel_offset, animate)


def load_frame_manifest(path):
    """Validate every encoded frame's index, sampling grid, and causal source."""
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty frame manifest: {path}")
    previous_source = -1
    previous_video = None
    for index, row in enumerate(rows):
        if row.get("frame_index") != index:
            raise ValueError("Frame manifest must enumerate contiguous encoded frames")
        source, video = row.get("physics_step"), row.get("video_physics_step")
        if not isinstance(source, int) or not isinstance(video, int):
            raise ValueError(
                "Frame manifest requires integer physics_step and video_physics_step"
            )
        if source < previous_source or source > video:
            raise ValueError(
                "Frame source ticks must be chronological and never in the future"
            )
        if previous_video is not None and video - previous_video != VIDEO_STRIDE:
            raise ValueError("Demo frame sampling must follow the 240 Hz / 30 fps grid")
        previous_source, previous_video = source, video
    return rows


def align_frame_manifests(ego, observer, *, seconds=None):
    """Return common ticks and contiguous source-video offsets.

    Different grid phases are rounded towards the past. Quantization is less
    than one video frame; capture gaps remain visible in the output manifest.
    """
    start = max(ego[0]["video_physics_step"], observer[0]["video_physics_step"])
    end = min(ego[-1]["video_physics_step"], observer[-1]["video_physics_step"])
    if seconds is not None:
        if seconds <= 0:
            raise ValueError("Preview seconds must be positive")
        end = min(end, start + (max(1, int(seconds * VIDEO_FPS)) - 1) * VIDEO_STRIDE)
    if end < start:
        raise ValueError("Camera recordings have no shared physics interval")
    offsets = [
        bisect_right([r["video_physics_step"] for r in rows], start) - 1
        for rows in (ego, observer)
    ]
    aligned = []
    for index, tick in enumerate(range(start, end + 1, VIDEO_STRIDE)):
        row = {"frame_index": index, "video_physics_step": tick}
        for name, rows, offset in zip(("ego", "observer"), (ego, observer), offsets):
            source = rows[offset + index]
            if not 0 <= tick - source["video_physics_step"] < VIDEO_STRIDE:
                raise ValueError("Input video grid cannot be aligned causally")
            row[name + "_frame_index"] = offset + index
            row[name + "_physics_step"] = source["physics_step"]
            row[name + "_video_physics_step"] = source["video_physics_step"]
        aligned.append(row)
    return aligned, offsets


def _human_state_label(state):
    """Describe the command phase only; never certify a rendered body pose."""
    animation = str(state.get("animation_state") or "unknown")
    active = state.get("active_action")
    status = str(state.get("action_status") or "unknown")
    if (
        animation.lower().replace("_", "") in ("lookaround", "lookingaround")
        and not active
    ):
        return "Look around (ambient)"
    if active:
        command = {"goto": "GoTo", "sit": "Sit"}.get(
            str(active).lower(), str(active).capitalize()
        )
        phase = {
            "walking_to_seat": "approach seat",
            "standing_up": "rising",
            "goto_pending": "pending",
        }.get(animation, animation.replace("_", " "))
        return f"{command} / {phase if status == 'ongoing' else status}"
    return animation.replace("_", " ")


def _human_animation_label(event):
    """Native clip-family evidence, not joint pose, blend weight or contact.

    A partial graph read cannot establish which other clips are active. Do
    not infer a pose from the command or carry a stale known state through a
    subsequent unavailable measurement.
    """
    state = event.get("animation_graph_state")
    families = event.get("active_clip_families")
    labels = {
        "idle": "Idle animation",
        "walk": "Walking animation",
        "sit": "Sitting animation",
        "stand": "Rising animation",
    }
    if (
        event.get("animation_graph_read_status") != "available"
        or not isinstance(families, list)
        or not all(isinstance(family, str) for family in families)
    ):
        return "Animation unknown"
    if state in labels and families == [state]:
        return labels[state]
    if (
        state == "mixed"
        and len(families) >= 2
        and len(set(families)) == len(families)
        and all(family in labels for family in families)
    ):
        return "Mixed animation"
    return "Animation unknown"


def _event_timeline(trace, heartbeats, human_replay_log):
    """Speech/outcomes lacking their own tick enter only at action-end evidence."""
    events = []
    # The command-application acknowledgment records when a message became
    # public. Prefer that instant to the later aggregate robot-action trace.
    # Bind each native speech occurrence to its exact authored trace turn.
    # A phase is evidence only when that occurrence has an explicit timing.
    turns = []
    for index, record in enumerate(trace):
        for raw in record.get("human_events", []):
            if raw.get("action") == "speak" and raw.get("message"):
                turns.append(
                    {
                        "index": index,
                        "step": record.get("step"),
                        "speaker": raw.get("human_id") or "Human",
                        "message": str(raw["message"]).strip(),
                        "timing": raw.get("timing"),
                    }
                )
    available = list(range(len(turns)))
    native_by_turn = {}
    speech_order = {}
    for event in human_replay_log:
        if (
            event.get("event")
            not in {"human_action_applied", "human_utterance_delivered"}
            or event.get("physics_step") is None
        ):
            continue
        message = str(event.get("message") or "").strip()
        human_id = event.get("human_id")
        if human_id and message:
            tick = int(event["physics_step"])
            candidates = [
                i
                for i in available
                if turns[i]["speaker"] == human_id and turns[i]["message"] == message
            ]
            if event.get("step") is not None:
                candidates = [
                    i for i in candidates if turns[i]["step"] == event["step"]
                ]
            match = candidates[0] if candidates else None
            phase = turns[match]["timing"] if match is not None else None
            events.append((tick, "speech", (human_id, message)))
            speech_order[id(events[-1])] = {"before_agent": 0, "after_agent": 2}.get(
                phase, 1
            )
            if match is not None:
                available.remove(match)
                native_by_turn[match] = tick
    for index, beat in enumerate(heartbeats[: len(trace)]):
        tick = beat.get("physics_steps")
        if tick is None:
            raise ValueError("Every action heartbeat requires physics_steps")
        robot_message = str(trace[index].get("robot_message") or "").strip()
        if robot_message:
            events.append((int(tick), "speech", ("Robot", robot_message)))
            speech_order[id(events[-1])] = 1
        for turn_index, turn in enumerate(turns):
            if turn["index"] != index or turn_index in native_by_turn:
                continue  # Native timestamps remain authoritative, including future ticks.
            events.append((int(tick), "speech", (turn["speaker"], turn["message"])))
            speech_order[id(events[-1])] = {"before_agent": 0, "after_agent": 2}.get(
                turn["timing"], 1
            )
        caption = execution_feedback_caption(trace[index])
        if caption:
            events.append((int(tick), "feedback", caption))
        for human_id, state in beat.get("humans", {}).items():
            events.append(
                (
                    int(tick),
                    "human_command",
                    (human_id, "phase", _human_state_label(state)),
                )
            )
    for event in human_replay_log:
        tick = event.get("physics_step")
        if tick is None:
            continue  # Decision-step-only history is not a physics timestamp.
        kind = event.get("event")
        action = event.get("action")
        command = event.get("command") or (
            action.get("command") if isinstance(action, dict) else None
        )
        human_id = event.get("human_id")
        if human_id and kind == "human_body_state_changed":
            events.append(
                (
                    int(tick),
                    "human_command",
                    (human_id, "phase", _human_state_label(event)),
                )
            )
        elif human_id and kind == "human_animation_graph_state_changed":
            events.append(
                (
                    int(tick),
                    "human_animation",
                    (human_id, _human_animation_label(event)),
                )
            )
        elif (
            human_id
            and command
            and kind in ("human_action_applied", "human_action_completed")
        ):
            status = "accepted" if kind == "human_action_applied" else "completed"
            command_label = {"goto": "GoTo", "sit": "Sit"}.get(
                str(command).lower(), str(command).capitalize()
            )
            events.append(
                (
                    int(tick),
                    "human_command",
                    (human_id, "ack", f"{command_label} {status}"),
                )
            )
        elif kind in ("milestone_event_occurred", "task_value_activated"):
            # A milestone event opens a phase; its completion is evaluated
            # separately. For example, passage_request_accepted can open
            # before the request has actually been answered.
            label = (
                "TASK UPDATE COMMITTED"
                if kind == "task_value_activated"
                else "PHASE AVAILABLE"
            )
            events.append(
                (
                    int(tick),
                    "feedback",
                    (label, str(event.get("milestone_id") or ""), "teal"),
                )
            )
    return sorted(events, key=lambda event: (event[0], speech_order.get(id(event), 1)))


def panel_states(physics_steps, trace, heartbeats, human_replay_log=()):
    """Compute only states supported by evidence available at each output tick."""
    if not trace or len(heartbeats) < len(trace):
        raise ValueError(
            "Demo requires a complete decision trace and action-end heartbeats"
        )
    indices = decision_indices_by_physics_step(physics_steps, trace, heartbeats)
    decisions = [_overlay_decision(record) for record in trace]
    if not any(decisions):
        raise ValueError("Demo trace has no current-schema model decisions")
    plans = _plan_states_by_trace_index(trace)
    events = _event_timeline(trace, heartbeats, human_replay_log)
    cursor = 0
    latest = {"speech": None, "feedback": None}
    human_commands, human_animations = {}, {}
    observed_command_phases = set()
    last_decision = None
    states = []
    for tick, index in zip(physics_steps, indices):
        while cursor < len(events) and events[cursor][0] <= tick:
            stamp, kind, value = events[cursor]
            if kind == "human_command":
                human_id, source, label = value
                if source == "phase":
                    observed_command_phases.add(human_id)
                if source == "phase" or human_id not in observed_command_phases:
                    human_commands[human_id] = label
            elif kind == "human_animation":
                human_animations[value[0]] = value[1]
            else:
                latest[kind] = (stamp, value)
            cursor += 1
        last_decision = decisions[index] or last_decision
        humans = []
        for human_id in sorted(human_commands.keys() | human_animations.keys()):
            command = human_commands.get(human_id)
            animation = human_animations.get(human_id)
            # Archives with no native graph evidence retain their historical
            # phase, explicitly identified as a command. Future graph events
            # do not change the labels at earlier ticks.
            label = (
                animation + (f" · Cmd: {command}" if command else "")
                if animation is not None
                else f"Command: {command}"
            )
            humans.append((human_id, label))
        state = {
            "decision": last_decision,
            "plan": plans[index],
            "archive_schema": bool(trace[index].get("presentation_legacy_schema")),
            "humans": tuple(humans),
        }
        for kind, record in latest.items():
            state[kind] = (
                record[1] if record and tick - record[0] < PHYSICS_HZ * 6 else None
            )
        states.append(state)
    return states


def write_demo_ass(
    *,
    path,
    physics_steps,
    trace,
    heartbeats,
    human_replay_log,
    layout,
    task_title,
    source_label,
    archive_preview=False,
    model_observations=None,
    grounding_depth_observations=None,
    recorded_result=None,
    terminal_frame_index=None,
):
    width, height = layout.resolution
    single_ego = isinstance(layout, SingleEgoLayout)
    ass = _BelowEgoAss(layout) if single_ego else _FocusAss(width, height)
    end = len(physics_steps) / VIDEO_FPS
    if not single_ego:
        ass.paragraph(
            0, end, 166, 17, source_label, 620, size=11, max_lines=1, color="muted"
        )
        ass.text(0, end, 24, 48, "EGO CAMERA", size=10, color="purple", bold=True)
        ass.text(
            0, end, 562, 48, "THIRD-PERSON CAMERA", size=10, color="teal", bold=True
        )
        mode = (
            "ARCHIVE LAYOUT PREVIEW · SYNC UNVERIFIED"
            if archive_preview
            else "RECORDED EXECUTION · SIMULATION TIME"
        )
        ass.text(0, end, 860, 18, mode, size=9.5, color="muted")
    ass.rule(0, end, 24, 462, 1232)
    if model_observations and grounding_depth_observations:
        raise ValueError("The shared right panel has exactly one evidence source")
    depth_panel = bool(grounding_depth_observations)
    sensor_observations = (
        grounding_depth_observations if depth_panel else model_observations
    )
    right_title = (
        "GROUNDING DEPTH · 640 × 480"
        if depth_panel
        else "MODEL OBSERVATION · 640 × 480"
        if model_observations
        else "OBSERVED SPATIAL MEMORY"
    )
    for x, title, color in [
        (24, "GOAL", "purple"),
        (406, "INTERACTION & EVENTS", "teal"),
        (866, right_title, "purple"),
    ]:
        ass.text(0, end, x, 482, title, size=10.5, color=color, bold=True)
    footer = (
        "DEBUG 640 PREVIEW"
        if single_ego and layout.width == 640
        else "NATIVE EGO · DISPLAY ONLY"
        if single_ego
        else "MODEL OBSERVATION: 640 × 480  /  PRESENTATION CAMERAS ARE DISPLAY ONLY"
    )
    if single_ego:
        footer += " / " + source_label
    ass.text(0, end, 24, 700, footer, size=9, color="muted")
    # Absolute simulation time keeps camera trims and edited excerpts auditable.
    for i in range(0, len(physics_steps), 3):
        stop = min(i + 3, len(physics_steps))
        seconds = physics_steps[i] / PHYSICS_HZ
        ass.text(
            i / VIDEO_FPS,
            stop / VIDEO_FPS,
            1112,
            700,
            f"SIM {seconds:07.2f} s",
            size=10,
            color="muted",
        )
    states = panel_states(physics_steps, trace, heartbeats, human_replay_log)
    if sensor_observations:
        indices = sensor_observation_indices(physics_steps, sensor_observations)
        first = 0
        while first < len(indices):
            last = first + 1
            while last < len(indices) and indices[last] == indices[first]:
                last += 1
            begin, finish = (
                (first * 100 // VIDEO_FPS) / 100,
                (last * 100 // VIDEO_FPS) / 100,
            )
            index = indices[first]
            if index < 0:
                ass.paragraph(
                    begin,
                    finish,
                    1110,
                    514,
                    "Awaiting recorded depth"
                    if depth_panel
                    else "Awaiting the first recorded model input",
                    146,
                    size=12,
                    max_lines=3,
                    color="muted",
                )
            else:
                capture = sensor_observations[index]
                tag = (
                    f"ACTION {capture.action_step:03d}"
                    if depth_panel
                    else f"REQUEST {capture.policy_step:03d}"
                )
                ass.text(
                    begin, finish, 1110, 514, tag, size=10.5, color="teal", bold=True
                )
                ass.paragraph(
                    begin,
                    finish,
                    1110,
                    541,
                    "Conditioned depth for grounding tools"
                    if depth_panel
                    else "Actual image sent to the model",
                    146,
                    size=12.5,
                    max_lines=3,
                )
                ass.text(
                    begin,
                    finish,
                    1110,
                    600,
                    f"CAPTURE {capture.physics_step / PHYSICS_HZ:.2f} s",
                    size=10,
                    color="muted",
                )
                ass.paragraph(
                    begin,
                    finish,
                    1110,
                    625,
                    "Recorded sensor view"
                    if depth_panel
                    else "Held until the next model observation",
                    146,
                    size=11,
                    max_lines=3,
                    color="muted",
                )
            first = last
    initial_goal = next(
        (s["plan"].goal for s in states if s["plan"] and s["plan"].goal), None
    )
    start = 0
    while start < len(states):
        stop = start + 1
        while stop < len(states) and states[stop] == states[start]:
            if recorded_result and stop == terminal_frame_index:
                break
            stop += 1
        # ASS has centisecond precision. Flooring to the immediately preceding
        # centisecond changes the intended 30 fps frame; rounding up would keep
        # the old state visible for one extra encoded frame at 2/30, 5/30, ... .
        begin = (start * 100 // VIDEO_FPS) / 100
        finish = (stop * 100 // VIDEO_FPS) / 100
        state = states[start]
        plan, decision = state["plan"], state["decision"]
        goal = plan.goal if plan and plan.goal else task_title
        if task_title and goal == initial_goal:
            goal = task_title
        # Long revised goals wrap before font size is reduced; never silently
        # hide a new recipient behind ellipsis. Editorial titles affect only
        # the initial goal, never a later model-authored goal revision.
        goal_size = 14.0
        while goal_size > 8 and any(
            line.endswith("…") for line in ass.wrap(goal, 350, goal_size, 3)
        ):
            goal_size -= 0.5
        lines = ass.wrap(goal, 350, goal_size, 3)
        if any(line.endswith("…") for line in lines):
            raise ValueError(
                "Goal is too long for the shared panel; provide a concise initial --task-title"
            )
        if len(lines) == 3:
            goal_size = min(goal_size, 11.5)
        ass.paragraph(begin, finish, 24, 502, goal, 350, size=goal_size, max_lines=3)
        if decision:
            active = min(max(decision.action_index, 0), len(decision.actions) - 1)
            label, detail = _action_caption(*decision.actions[active])
            current = _current_task_text(decision, plan, state["archive_schema"])
            ass.text(
                begin,
                finish,
                24,
                548,
                "CURRENT TASK",
                size=10,
                color="purple",
                bold=True,
            )
            ass.paragraph(begin, finish, 24, 565, current, 350, size=12.5, max_lines=2)
            ass.text(begin, finish, 24, 608, "NOW", size=10, color="purple", bold=True)
            ass.paragraph(
                begin,
                finish,
                65,
                603,
                label + (" · " + detail if detail else ""),
                309,
                size=16.5,
                max_lines=1,
                bold=True,
            )
            ass.text(
                begin, finish, 24, 642, "REASONING", size=10, color="purple", bold=True
            )
            ass.paragraph(
                begin,
                finish,
                24,
                659,
                decision.thinking,
                350,
                size=11.5,
                max_lines=2,
                color="muted",
            )
        speech = state["speech"]
        if speech:
            ass.text(
                begin,
                finish,
                406,
                504,
                speech[0].upper(),
                size=10,
                color="teal",
                bold=True,
            )
            ass.paragraph(begin, finish, 406, 524, speech[1], 424, size=15, max_lines=3)
        else:
            ass.text(
                begin,
                finish,
                406,
                504,
                "No recent spoken interaction",
                size=12,
                color="muted",
            )
        feedback = state["feedback"]
        if (
            recorded_result
            and terminal_frame_index is not None
            and start >= terminal_frame_index
        ):
            ass.text(
                begin,
                finish,
                406,
                590,
                "RECORDED RESULT",
                size=9.5,
                color="teal",
                bold=True,
            )
            summary = (
                "Task completed"
                if recorded_result.get("success") is True
                else str(
                    recorded_result.get("termination_reason")
                    or recorded_result.get("outcome_status")
                )
            )
            ass.paragraph(begin, finish, 406, 609, summary, 424, size=14, max_lines=2)
        elif feedback:
            label, detail, color = feedback
            ass.text(
                begin,
                finish,
                406,
                590,
                "PREVIOUS FEEDBACK" if single_ego else label,
                size=9.5,
                color=color,
                bold=True,
            )
            ass.paragraph(
                begin,
                finish,
                406,
                609,
                detail,
                424,
                size=12,
                max_lines=2,
                color="muted",
            )
        human_text = "  ·  ".join(
            f"{name}: {status}" for name, status in state["humans"]
        )
        if human_text:
            ass.paragraph(
                begin,
                finish,
                406,
                652,
                human_text,
                424,
                size=10.5,
                max_lines=2,
                color="muted",
            )
        start = stop
    path = Path(path)
    path.write_text(ass.header + "\n".join(ass.events) + "\n")
    return path


def probe_video(path, ffprobe="ffprobe"):
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    return {
        "resolution": (int(stream["width"]), int(stream["height"])),
        "fps": numerator / denominator,
        "frame_count": int(stream["nb_frames"]),
    }


def load_display_affines(
    path, *, ego_path, ego_manifest, source_steps, ffprobe="ffprobe"
):
    """Bind a derived display video to exact original frames; camera evidence stays raw."""
    import math
    import numpy as np
    from .demo_stabilization import (
        StabilizationSettings,
        native_affine,
        covers_native_canvas,
    )

    proof = json.loads(Path(path).read_bytes())
    if proof.get("schema_version") != "tarsphere.demo.ego_stabilization.v1":
        raise ValueError("Unsupported display stabilization sidecar")
    estimate_path = Path(proof["estimate"])
    if file_sha256(estimate_path) != proof["estimate_sha256"]:
        raise ValueError("Display affine estimate hash mismatch")
    plan = json.loads(estimate_path.read_bytes())
    if (
        Path(plan["source_video"]).resolve() != Path(ego_path).resolve()
        or Path(plan["source_frame_manifest"]).resolve() != Path(ego_manifest).resolve()
        or plan["source_video_sha256"] != file_sha256(ego_path)
        or plan["source_frame_manifest_sha256"] != file_sha256(ego_manifest)
        or proof["source_video_sha256"] != plan["source_video_sha256"]
        or proof["source_frame_manifest_sha256"] != plan["source_frame_manifest_sha256"]
        or proof["source_video"]
        != {"path": plan["source_video"], "sha256": plan["source_video_sha256"]}
        or proof["source_frame_manifest"]
        != {
            "path": plan["source_frame_manifest"],
            "sha256": plan["source_frame_manifest_sha256"],
        }
        or plan["source_start_frame"] != 0
        or plan["is_source_excerpt"]
        or proof["source_start_frame"] != 0
        or proof["is_source_excerpt"]
        or plan["frame_count"] != len(source_steps)
        or proof["frame_count"] != len(source_steps)
        or plan["fps"] != "30/1"
        or proof["fps"] != "30/1"
        or proof.get("source_clock_unchanged") is not True
        or proof.get("HUD_or_map_warped") is not False
    ):
        raise ValueError(
            "Display affine source/clock differs from the complete original take"
        )
    affine_path = Path(plan["affine_jsonl"])
    if (
        file_sha256(affine_path) != plan["affine_jsonl_sha256"]
        or proof["affine_jsonl_sha256"] != plan["affine_jsonl_sha256"]
        or proof["affine_jsonl"]
        != {"path": str(affine_path), "sha256": plan["affine_jsonl_sha256"]}
    ):
        raise ValueError("Display affine rows hash mismatch")
    rows = [
        json.loads(line)
        for line in affine_path.read_text().splitlines()
        if line.strip()
    ]
    settings = dict(plan["settings"])
    for key in ("native_size", "working_size"):
        settings[key] = tuple(settings[key])
    settings = StabilizationSettings(**settings).validate()
    if len(rows) != len(source_steps):
        raise ValueError("Display affine count differs from the source clock")
    matrices = []
    source_rows = load_frame_manifest(ego_manifest)
    for index, (row, tick) in enumerate(zip(rows, source_steps)):
        matrix = np.asarray(
            row["affine_source_native_to_stabilized_native"], dtype=float
        )
        expected = native_affine(
            row["correction_native_shift_px"],
            math.radians(row["correction_roll_deg"]),
            settings,
        )
        if (
            row["frame_index"] != index
            or row["source_frame_index"] != index
            or row["physics_step"] != tick
            or row["source_physics_step"] != tick
            or row["source_video_physics_step"]
            != source_rows[index].get("video_physics_step", tick)
            or np.linalg.norm(row["correction_native_shift_px"])
            > settings.max_shift_px + 1e-6
            or abs(row["correction_roll_deg"]) > settings.max_roll_deg + 1e-6
            or row["fixed_crop_per_edge"] != settings.crop_per_edge
            or row["canvas_fully_source_supported"] is not True
            or matrix.shape != (2, 3)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, expected, atol=1e-8, rtol=0)
            or not covers_native_canvas(matrix, (2560, 1920))
        ):
            raise ValueError(
                "Display affine frame identity or bounded native geometry mismatch"
            )
        matrices.append(matrix.tolist())
        if (
            index
            and source_steps[index] == source_steps[index - 1]
            and settings.preset == "gimbal"
            and not np.array_equal(matrix, np.asarray(matrices[index - 1]))
        ):
            raise ValueError(
                "Gimbal duplicate capture tick must retain the same affine"
            )
    videos = [
        entry for entry in proof["outputs"] if entry["probe"].get("width") == 2560
    ]
    if len(videos) != 1:
        raise ValueError("Display affine requires exactly one native derived video")
    video = videos[0]
    derived = Path(video["path"])
    if (
        proof["derived_video"] != {"path": str(derived), "sha256": video["sha256"]}
        or derived.resolve() == Path(ego_path).resolve()
        or file_sha256(derived) != video["sha256"]
        or probe_video(derived, ffprobe)
        != {"resolution": (2560, 1920), "fps": 30.0, "frame_count": len(source_steps)}
    ):
        raise ValueError("Derived display video hash/dimensions/frame count mismatch")
    return (
        derived,
        matrices,
        dict(
            schema_version="tarsphere.demo.display_stabilization_binding.v1",
            sidecar=str(Path(path).resolve()),
            sidecar_sha256=file_sha256(path),
            estimate=str(estimate_path.resolve()),
            estimate_sha256=proof["estimate_sha256"],
            source_video=str(Path(ego_path).resolve()),
            source_video_sha256=plan["source_video_sha256"],
            source_frame_manifest_sha256=plan["source_frame_manifest_sha256"],
            display_video=str(derived.resolve()),
            display_video_sha256=video["sha256"],
            affine_jsonl=str(affine_path.resolve()),
            affine_jsonl_sha256=plan["affine_jsonl_sha256"],
            frame_count=len(rows),
            settings=plan["settings"],
            scope="source native ego warp before static HUD/map; ABC points use the identical2D affine",
            measured_camera_pose=False,
            original_camera_and_clock_unchanged=True,
        ),
    )


def render_demo_presentation(
    *,
    output_path,
    ego_path,
    ego_manifest,
    observer_path,
    observer_manifest,
    trace,
    heartbeats,
    online_map_dir,
    human_replay_log=(),
    task_title="",
    source_label="",
    height=1440,
    seconds=None,
    archive_preview=False,
    show_model_observation=False,
    model_observations=None,
    show_grounding_depth=False,
    grounding_depth_observations=None,
    ffmpeg="ffmpeg",
    single_ego=False,
    debug_preview=False,
    terminal_hold_seconds=0.0,
    recorded_result=None,
    full_hud=False,
    display_grade="none",
    lidar_observations=None,
    annotations=None,
    preserve_map=False,
    ego_frontier_labels=False,
    frontier_captures=None,
    frontier_model_source=None,
    ego_frontier_cache=None,
    display_affine_manifest=None,
    policy_ego_evidence=None,
):
    """Encode synchronized clean cameras and a single evidence panel below them.

    Archive preview is explicitly a layout study when the third camera lacks a
    timestamp manifest. It cannot satisfy native-resolution or sync acceptance.
    """
    from .display_color_grade import resolve_display_grade, display_grade_metadata

    grade_filter = resolve_display_grade(display_grade)
    if policy_ego_evidence and (
        not full_hud
        or not single_ego
        or not preserve_map
        or debug_preview
        or archive_preview
        or grade_filter
        or display_affine_manifest
        or ego_frontier_cache
    ):
        raise ValueError(
            "Actual policy ego requires full HUD/map and original pixels without grading, stabilization, or a native frontier cache"
        )
    if preserve_map and not full_hud:
        raise ValueError("Preserving the original map requires full HUD")
    if ego_frontier_labels and (
        not preserve_map or debug_preview or not frontier_captures
    ):
        raise ValueError(
            "Ego frontier labels require native full HUD, preserved map and recorded captures"
        )
    if ego_frontier_cache and not ego_frontier_labels:
        raise ValueError("A reviewed frontier cache requires ego frontier labels")
    if display_affine_manifest and (not full_hud or debug_preview or archive_preview):
        raise ValueError("Display affine requires verified native full HUD")
    if display_affine_manifest and ego_frontier_labels and not ego_frontier_cache:
        raise ValueError(
            "Stabilized ego labels require a reviewed raw-coordinate cache"
        )
    if grade_filter and not single_ego:
        raise ValueError("Display grade currently requires single ego layout")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Preserve existing rendered take: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if full_hud and not single_ego:
        raise ValueError("Full HUD requires single ego layout")
    if annotations and (not full_hud or archive_preview):
        raise ValueError("Editorial annotations require the verified full HUD layout")
    if lidar_observations and not full_hud:
        raise ValueError("Recorded LiDAR currently requires full HUD layout")
    layout = None if single_ego else DemoLayout(height)
    ego = load_frame_manifest(ego_manifest)
    probe = str(Path(ffmpeg).with_name("ffprobe"))
    if terminal_hold_seconds < 0 or (terminal_hold_seconds and not single_ego):
        raise ValueError(
            "Terminal editorial hold requires single ego layout and nonnegative seconds"
        )
    if single_ego and (
        observer_path is not None or observer_manifest is not None or archive_preview
    ):
        raise ValueError("Single ego layout does not accept observer or archive inputs")
    if debug_preview and not single_ego:
        raise ValueError("Debug640 preview requires single ego layout")
    inputs = [ego_path] if single_ego else [ego_path, observer_path]
    streams = [probe_video(path, probe) for path in inputs]
    if len(ego) != streams[0]["frame_count"]:
        raise ValueError("Ego frame manifest count disagrees with the video")
    display_ego_path, display_affines, display_stabilization = ego_path, None, None
    if display_affine_manifest:
        display_ego_path, display_affines, display_stabilization = load_display_affines(
            display_affine_manifest,
            ego_path=ego_path,
            ego_manifest=ego_manifest,
            source_steps=[row["physics_step"] for row in ego],
            ffprobe=probe,
        )
    if single_ego:
        expected = (640, 480) if debug_preview or policy_ego_evidence else (2560, 1920)
        if streams[0]["resolution"] != expected or streams[0]["fps"] != VIDEO_FPS:
            raise ValueError(
                f"Single ego requires native {expected} at 30 fps, got {streams[0]}"
            )
        if full_hud:
            from .demo_full_hud import FullHudLayout

            layout = FullHudLayout(*((2560, 1920) if policy_ego_evidence else expected))
        else:
            layout = SingleEgoLayout(*expected)
        if seconds is not None and seconds <= 0:
            raise ValueError("Preview seconds must be positive")
        count = (
            len(ego)
            if seconds is None
            else min(len(ego), max(1, int(seconds * VIDEO_FPS)))
        )
        aligned = [
            {
                "frame_index": i,
                "video_physics_step": row["video_physics_step"],
                "ego_frame_index": i,
                "ego_physics_step": row["physics_step"],
            }
            for i, row in enumerate(ego[:count])
        ]
        offsets = [0]
    elif archive_preview:
        if observer_manifest is not None:
            raise ValueError(
                "Archive layout mode requires absent observer timing; use normal mode for synchronized recordings"
            )
        count = min(
            len(ego), int(streams[1]["frame_count"] / streams[1]["fps"] * VIDEO_FPS)
        )
        if seconds is not None:
            if seconds <= 0:
                raise ValueError("Preview seconds must be positive")
            count = min(count, max(1, int(seconds * VIDEO_FPS)))
        aligned = [
            {
                "frame_index": i,
                "video_physics_step": row["video_physics_step"],
                "ego_frame_index": i,
                "ego_physics_step": row["physics_step"],
                "observer_physics_step": None,
            }
            for i, row in enumerate(ego[:count])
        ]
        offsets = [0, 0]
    else:
        for name, stream, expected in zip(
            ("ego", "observer"), streams, ((2560, 1920), (2560, 1440))
        ):
            if stream["resolution"] != expected or stream["fps"] != VIDEO_FPS:
                raise ValueError(
                    f"{name} requires native {expected} at 30 fps, got {stream}"
                )
        observer = load_frame_manifest(observer_manifest)
        if len(observer) != streams[1]["frame_count"]:
            raise ValueError("Observer frame manifest count disagrees with the video")
        aligned, offsets = align_frame_manifests(ego, observer, seconds=seconds)
    if not aligned:
        raise ValueError("No video frames available to render")
    source_frame_count = len(aligned)
    hold_frames = round(terminal_hold_seconds * VIDEO_FPS)
    if hold_frames and source_frame_count != len(ego):
        raise ValueError(
            "Terminal editorial hold requires the complete source recording"
        )
    if hold_frames and not recorded_result:
        raise ValueError("Terminal editorial hold requires the recorded result")
    final_source = aligned[-1]
    for index in range(hold_frames):
        aligned.append(
            {
                **final_source,
                "frame_index": source_frame_count + index,
                "editorial_hold": True,
                "presentation_time_s": (source_frame_count + index) / VIDEO_FPS,
            }
        )
    steps = [
        row["ego_physics_step"] if single_ego else row["video_physics_step"]
        for row in aligned
    ]
    count = len(steps)
    if show_model_observation != bool(model_observations):
        raise ValueError(
            "Model-observation panel requires its verified request-image evidence"
        )
    if show_grounding_depth != bool(grounding_depth_observations):
        raise ValueError(
            "Grounding-depth panel requires verified conditioned sensor evidence"
        )
    if show_model_observation and show_grounding_depth:
        raise ValueError("The shared right panel has exactly one evidence source")
    recorded_sensor_condition = (
        "grounding_depth"
        if show_grounding_depth
        else "model_observation"
        if show_model_observation
        else "observed_spatial_memory"
    )
    right_panel = (
        "observed_spatial_memory" if preserve_map else recorded_sensor_condition
    )
    map_path = output.with_suffix(
        ".map.mp4"
        if preserve_map
        else ".grounding-depth.mp4"
        if show_grounding_depth
        else ".model-observation.mp4"
        if show_model_observation
        else ".map.mp4"
    )
    root = Path(online_map_dir)
    rgbd = root / "presentation" / "rgbd"
    sensor_card = None
    if (show_model_observation or show_grounding_depth) and not preserve_map:
        sensor_options = {}
        if full_hud:
            mx, my, _, _ = layout.rect("map")
            ix, iy, iw, ih = layout.rect("sensor_image")
            sensor_options["image_rect"] = (ix - mx, iy - my, iw, ih)
            # Presentation metadata describes the display, not a new sensor capture.
            sensor_card = {
                "title": "CONDITIONED DEPTH"
                if show_grounding_depth
                else "MODEL OBSERVATION",
                "source_resolution": [640, 480],
                "image_rect": [ix, iy, iw, ih],
                "fit": "contain",
                "complete_source_view": True,
                "scope": "Conditioned depth for grounding tools"
                if show_grounding_depth
                else "Actual image sent to the model",
                "clock": "measured pre-agent camera_capture.physics_step / 240",
                "display": "latest capture at or before source frame; frozen with terminal source hold",
                "identity": "action_step" if show_grounding_depth else "policy_step",
            }
        write_sensor_observation_video(
            output_path=map_path,
            physics_steps=steps,
            observations=grounding_depth_observations
            if show_grounding_depth
            else model_observations,
            resolution=layout.rect("map")[2:],
            **sensor_options,
        )
        map_kind = None
    elif single_ego:
        write_online_map_video(
            output_path=map_path,
            online_map_dir=root,
            physics_steps=steps,
            observer_fps=VIDEO_FPS,
            trace=trace,
            heartbeats=heartbeats,
            resolution=layout.rect("map")[2:],
            paper=full_hud,
            **({"native_frontier_badges": False} if preserve_map else {}),
        )
        map_kind = "model_visible_exploration_memory"
    elif any(rgbd.glob("*.npz")):
        from tarsphere.infra.perception.presentation_rgbd import write_layered_map_video

        write_layered_map_video(
            output_path=map_path,
            snapshot_dir=rgbd,
            physics_steps=steps,
            observer_fps=VIDEO_FPS,
            resolution=layout.rect("map")[2:],
        )
        map_kind = "measured_rgbd_and_online_occupancy"
    elif archive_preview:
        write_online_map_video(
            output_path=map_path,
            online_map_dir=root,
            physics_steps=steps,
            observer_fps=VIDEO_FPS,
            trace=trace,
            heartbeats=heartbeats,
            resolution=layout.rect("map")[2:],
        )
        map_kind = "archived_online_map"
    else:
        raise FileNotFoundError(
            f"Publication demo requires measured RGB-D map snapshots: {rgbd}"
        )
    ass_writer = write_demo_ass
    annotation_report = []
    if full_hud:
        from .demo_full_hud import write_full_hud_ass

        states = (
            annotated_panel_states(
                steps, trace, heartbeats, human_replay_log, annotations
            )
            if annotations
            else panel_states(steps, trace, heartbeats, human_replay_log)
        )
        ass_path = write_full_hud_ass(
            path=output.with_suffix(".ass"),
            layout=layout,
            states=states,
            task_title=task_title,
            source_label=source_label,
            recorded_result=recorded_result,
            physics_steps=steps,
            lidar_observations=lidar_observations,
            model_observations=None if preserve_map else model_observations,
            grounding_depth_observations=None
            if preserve_map
            else grounding_depth_observations,
            **({"annotation_report": annotation_report} if annotations else {}),
            terminal_frame_index=source_frame_count - 1
            if source_frame_count == len(ego)
            else None,
        )
    else:
        ass_path = ass_writer(
            path=output.with_suffix(".ass"),
            physics_steps=steps,
            trace=trace,
            heartbeats=heartbeats,
            human_replay_log=human_replay_log,
            layout=layout,
            task_title=task_title,
            source_label=source_label,
            archive_preview=archive_preview,
            model_observations=model_observations,
            grounding_depth_observations=grounding_depth_observations,
            recorded_result=recorded_result if single_ego else None,
            terminal_frame_index=source_frame_count - 1
            if source_frame_count == len(ego)
            else None,
        )
    if policy_ego_evidence:
        from .demo_full_hud import _PaperHudAss

        badge = _PaperHudAss(*layout.resolution)
        condition = policy_ego_evidence["condition"]
        label = "ROBOT INPUT · 640 × 480"
        if condition == "reduced_resolution":
            sw, sh = policy_ego_evidence["effective_sensor_resolution"]
            label += f" · Sensor {sw} × {sh}"
        elif condition == "low_light":
            label += f" · Low light ({policy_ego_evidence['low_light_exposure_scale']:g}× exposure)"
        else:
            label += " · Conditioned depth"
        badge.box(0, count / VIDEO_FPS, 16, 678, 650, 26, color="ink")
        badge.text(
            0, count / VIDEO_FPS, 24, 682, label, size=14, color="white", bold=True
        )
        with ass_path.open("a") as handle:
            handle.write("\n" + "\n".join(badge.events) + "\n")
    frontier_report = None
    if ego_frontier_labels:
        from .ego_frontier_presentation import (
            append_ego_frontier_labels,
            append_cached_ego_frontier_labels,
        )

        frontier_report = (
            append_cached_ego_frontier_labels(
                cache_path=ego_frontier_cache,
                ass_path=ass_path,
                ego_path=ego_path,
                source_steps=steps[:source_frame_count],
                output_steps=steps,
                trace=trace,
                heartbeats=heartbeats,
                resolution=layout.resolution,
                model_source=frontier_model_source,
                display_affines=display_affines,
            )
            if ego_frontier_cache
            else append_ego_frontier_labels(
                ass_path=ass_path,
                ego_path=ego_path,
                source_steps=steps[:source_frame_count],
                output_steps=steps,
                trace=trace,
                heartbeats=heartbeats,
                online_map_dir=root,
                captures=frontier_captures,
                resolution=layout.resolution,
                ffmpeg=ffmpeg,
                model_source=frontier_model_source,
            )
        )
    width, height = layout.resolution
    ex, ey, ew, eh = layout.rect("ego")
    if not single_ego:
        ox, oy, ow, oh = layout.rect("observer")
    mx, my, mw, mh = layout.rect("map")
    if single_ego:
        # The ego source keeps native dimensions without scale or crop.
        # Below-panel UI stays outside ego; full HUD deliberately overlays its cards.
        # Optional display grading changes only ego colours before map/UI composition.
        filters = [
            f"[0:v]trim=end_frame={source_frame_count},setpts=PTS-STARTPTS,setsar=1,"
            f"tpad=stop_mode=clone:stop={hold_frames},"
            + (f"scale={ew}:{eh}:flags=neighbor," if policy_ego_evidence else "")
            + (grade_filter + "," if grade_filter else "")
            + f"pad={width}:{height}:0:0:color={'0xF9F7FB' if full_hud else '0x15171E'}[canvas]",
            f"[canvas][1:v]overlay={mx}:{my}:shortest=1,"
            f"{ffmpeg_subtitles_filter(ass_path)}[out]",
        ]
        command = [ffmpeg, "-y", "-i", str(display_ego_path), "-i", str(map_path)]
        if full_hud:
            q = height / 720
            logo = (
                Path(__file__).resolve().parents[3]
                / "assets/presentation/tarsphere-logo.png"
            )
            lx, ly = round(ew + 28 * q), round(628 * q)
            filters[-1] = filters[-1].replace("[out]", "[hud]")
            filters += [
                f"[hud]drawbox=x={lx}:y={ly}:w={round(164 * q)}:h={round(45 * q)}:color=0xFFFFFF:t=fill[brand]",
                f"[2:v]crop=1805:495:116:149,scale={round(164 * q)}:-1:flags=lanczos[logo]",
                f"[brand][logo]overlay={lx}:{ly}:eof_action=repeat[out]",
            ]
            command += ["-i", str(logo)]
    else:
        filters = []
        for index, (offset, w, h) in enumerate(
            ((offsets[0], ew, eh), (offsets[1], ow, oh))
        ):
            # fps only repeats the explicitly labeled old 5 fps layout-study footage.
            rate = f"fps={VIDEO_FPS}," if archive_preview and index == 1 else ""
            filters.append(
                f"[{index}:v]{rate}trim=start_frame={offset}:end_frame={offset + count},"
                f"setpts=PTS-STARTPTS,scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x101219,setsar=1[v{index}]"
            )
        q = height / 720
        logo = (
            Path(__file__).resolve().parents[3]
            / "assets/presentation/tarsphere-logo.png"
        )
        if not logo.is_file():
            raise FileNotFoundError(f"Original TARSphere logo missing: {logo}")
        filters += [
            f"[v0]pad={width}:{height}:{ex}:{ey}:color=0x15171E[canvas]",
            f"[canvas][v1]overlay={ox}:{oy}:shortest=1[pair]",
            f"[pair][2:v]overlay={mx}:{my}:shortest=1,"
            f"drawbox=x={round(24 * q)}:y={round(6 * q)}:w={round(128 * q)}:h={round(35 * q)}:"
            f"color=0xF2F0F5:t=fill[brand]",
            f"[3:v]crop=1805:495:116:149,scale={round(120 * q)}:-1:flags=lanczos[logo]",
            f"[brand][logo]overlay={round(28 * q)}:{round(7 * q)}:eof_action=repeat,"
            f"{ffmpeg_subtitles_filter(ass_path)}[out]",
        ]
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(ego_path),
            "-i",
            str(observer_path),
            "-i",
            str(map_path),
            "-i",
            str(logo),
        ]
    result = subprocess.run(
        command
        + [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-filter_complex_threads",
            "2",
            "-threads",
            "4",
            "-frames:v",
            str(count),
            "-r",
            str(VIDEO_FPS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "16" if full_hud else "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode or not output.is_file():
        raise RuntimeError("Demo composition failed: " + result.stderr[-1800:])
    final_stream = probe_video(output, probe)
    if (
        final_stream["resolution"] != layout.resolution
        or final_stream["frame_count"] != count
    ):
        raise RuntimeError(
            f"Demo output failed frame/dimension verification: {final_stream}"
        )
    manifest = output.with_suffix(".frames.jsonl")
    manifest.write_text("".join(json.dumps(row) + "\n" for row in aligned))
    metadata = {
        "video": str(output.resolve()),
        "output_resolution": list(layout.resolution),
        "fps": VIDEO_FPS,
        "physics_hz": PHYSICS_HZ,
        "frame_count": count,
        "first_physics_step": steps[0],
        "last_physics_step": steps[-1],
        "archive_layout_preview": archive_preview,
        "synchronization_verified": not archive_preview,
        "initial_goal_editorial_title": task_title,
        "recorded_model_label": source_label,
        "layout": "single_ego_full_hud"
        if full_hud
        else "single_ego_below_panels"
        if single_ego
        else "dual_camera",
        "visual_theme": "paper_reference_light" if full_hud else "original",
        "hud_text_rendering": {
            "method": "native-resolution ASS vector text",
            "resolution": list(layout.resolution),
            "font_family": "Times New Roman",
            "intermediate_text_raster": False,
            "h264_crf": 16,
            "font_scope": "generated HUD/radar/map frame and legend/native ego badges; original memory PNG internal labels remain source raster; brand logo retained",
            "logical_grid": {
                "height": 720,
                "ego_width": 960,
                "task": [16, 16, 324, 132],
                "dialogue": [340, 16, 460, 132],
                "lidar": [800, 16, 144, 132],
                "header_height": 36,
                "top": 16,
                "header_bottom": 52,
                "bottom": 148,
                "corner_radius": 0,
                "alpha_hex": "70",
                "header_alpha_hex": "70",
            },
            "top_body_style": {
                "fill_rgb": "#F3EEF7",
                "opacity": 1 - 112 / 255,
                "text_primary_rgb": "#322A30",
                "text_secondary_rgb": "#322A30",
                "text_outline_logical_px": 0,
            },
        }
        if full_hud
        else None,
        "recorded_lidar": {
            "enabled": bool(lidar_observations),
            "source": "hash-bound original policy request text",
            "clock": "measured pre-agent camera_capture.physics_step",
            "display": "latest recorded scan at or before each source frame; sparse returns only",
            "scan_count": len(lidar_observations or []),
            "observations": [
                {key: value for key, value in scan.items() if key != "scan"}
                for scan in lidar_observations or []
            ],
        },
        "debug_preview": debug_preview,
        "display_grade": display_grade_metadata(display_grade),
        "editorial_terminal_hold": {
            "frames": hold_frames,
            "seconds": hold_frames / VIDEO_FPS,
            "frozen_source_physics_step": final_source["ego_physics_step"],
            "frozen_source_video_physics_step": final_source["video_physics_step"],
            "source_frame_index": final_source["ego_frame_index"],
        },
        "recorded_result": recorded_result if single_ego else None,
        "ego_rect": list(layout.rect("ego")),
        "input_videos": [str(Path(p).resolve()) for p in inputs],
        "input_video_sha256": [file_sha256(p) for p in inputs],
        "input_frame_manifests": [
            str(Path(ego_manifest).resolve()),
            *(
                [str(Path(observer_manifest).resolve()) if observer_manifest else None]
                if not single_ego
                else []
            ),
        ],
        "input_frame_manifest_sha256": [
            file_sha256(ego_manifest),
            *(
                [file_sha256(observer_manifest) if observer_manifest else None]
                if not single_ego
                else []
            ),
        ],
        "source_streams": streams,
        "map_kind": map_kind,
        "right_panel": right_panel,
        "model_observation_evidence": [
            capture.provenance() for capture in (model_observations or [])
        ],
        "grounding_depth_evidence": [
            capture.provenance() for capture in (grounding_depth_observations or [])
        ],
        "frame_manifest": str(manifest.resolve()),
        "subtitle_file": str(ass_path.resolve()),
        "model_observation_resolution": [640, 480],
        "policy_ego": (
            {
                **policy_ego_evidence,
                "display_enlargement": "nearest-neighbour pixel replication for separately rendered high-resolution HUD",
                "native_2k_presentation_camera_used": False,
                "display_stabilized": False,
                "display_colour_graded": False,
                "super_resolution": False,
            }
            if policy_ego_evidence
            else None
        ),
        "video_sha256": file_sha256(output),
        "frame_manifest_sha256": file_sha256(manifest),
    }
    if annotations:
        metadata["editorial_annotation_visibility"] = annotation_report
    if preserve_map:
        metadata["recorded_sensor_condition"] = recorded_sensor_condition
        metadata["source_map_display"] = {
            "mode": "complete_recorded_memory_png",
            "extra_bev_frontier_badges": False,
            "frontier_reprojection": False,
            "original_small_labels_preserved": True,
            "display_style": "paper frame and display palette; original PNG bytes unchanged",
            "clock": "original decision and source physics clock",
        }
    if frontier_report is not None:
        if policy_ego_evidence:
            frontier_report["display_source"] = (
                "original conditioned 640x480 robot sensor, with fourfold pixel display"
            )
            frontier_report["independent_2k_camera_used"] = False
        metadata["ego_frontier_labels"] = frontier_report
    if display_stabilization is not None:
        metadata["display_stabilization"] = display_stabilization
    if sensor_card is not None:
        metadata["sensor_card"] = sensor_card
    output.with_suffix(".presentation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata
