"""Preserve actual conditioned depth views before temporary capture cleanup."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image


def preserve_conditioned_depth_captures(steps, *, output_dir, capture_root):
    """Copy clock-bound runtime RGB|depth PNGs; crop their actual right half.

    These are the conditioned sensor visualizations used alongside grounding,
    not images sent to the language model and not metric-depth arrays. No new
    sensor capture, noise generation, rescaling, or simulation step occurs.
    """
    output_dir, capture_root = Path(output_dir), Path(capture_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records, captures, previous_tick = [], {}, -1
    for value in steps:
        row = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        observation = row["observation"]
        clock = observation.get("camera_capture") or {}
        if (
            observation.get("timing") != "pre_agent"
            or observation.get("step") != row["decision_step"]
            or any(
                type(clock.get(key)) is not int or clock[key] < 0
                for key in ("physics_step", "capture_id")
            )
        ):
            raise ValueError(
                "Conditioned depth requires the actual pre-agent capture clock."
            )
        tick, capture = clock["physics_step"], clock["capture_id"]
        if tick < previous_tick:
            raise ValueError("Conditioned depth capture clocks are not chronological.")
        previous_tick = tick
        source = Path(observation["image_path"]).resolve()
        if not source.is_relative_to(capture_root):
            raise ValueError(
                "Conditioned depth source escapes this live capture directory."
            )
        data = source.read_bytes()
        source_sha = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (1280, 480) or image.mode != "RGB":
                raise ValueError(
                    "Conditioned depth requires the native 640 RGB | 640 depth PNG."
                )
            depth = image.crop((640, 0, 1280, 480))
        key = (tick, capture)
        if key in captures:
            if captures[key]["source_rgbd_sha256"] != source_sha:
                raise ValueError("One capture identity has conflicting sensor pixels.")
            evidence = captures[key]
        else:
            name = f"step_{tick:06d}_capture_{capture:06d}"
            rgbd_path, depth_path = (
                output_dir / (name + ".rgbd.png"),
                output_dir / (name + ".depth.png"),
            )
            rgbd_path.write_bytes(data)
            depth.save(depth_path)
            evidence = {
                "physics_step": tick,
                "capture_id": capture,
                "source_rgbd_path": str(rgbd_path),
                "source_rgbd_sha256": source_sha,
                "depth_view_path": str(depth_path),
                "depth_view_sha256": hashlib.sha256(
                    depth_path.read_bytes()
                ).hexdigest(),
                "depth_resolution": [640, 480],
                "crop_box": [640, 0, 1280, 480],
                "kind": "runtime_conditioned_depth_visualization",
                "scope": "Actual runtime RGBD right half; tool-private sensor evidence, not a provider image or metric depth array.",
            }
            captures[key] = evidence
        records.append(
            {
                **evidence,
                "action_step": row["decision_step"],
                "policy_inference": row["policy_inference"],
                "observation_id": observation["id"],
            }
        )
    if not records:
        raise ValueError("No actual conditioned depth captures were recorded.")
    return records
