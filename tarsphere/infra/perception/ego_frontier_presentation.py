"""Offline native ego labels: captured camera seeds and conservative image tracking.

No video, model input, camera state or simulation clock is modified. Between
captured seeds this is image-space correspondence, never measured 3D projection.
"""

from pathlib import Path
import hashlib
import json
import math
import re
import subprocess

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captured_seed(path):
    """Inverse the frozen 640 -> 439x329 -> (60,4) FrontierNet crop.

    Independently check the inverse against original world points and the exact
    observation's graph camera transform. Missing/contradictory evidence fails.
    """
    path = Path(path)
    raw = json.loads(path.read_bytes())
    graph_path = path.with_name(path.name.replace("_candidates.json", "_graph.json"))
    graph = json.loads(graph_path.read_bytes())
    step = int(raw["observation_step"])
    poses = [p for p in graph["poses"] if int(p["observation_step"]) == step]
    if len(poses) != 1:
        raise ValueError("Ego seed requires one exact observation camera transform")
    transform = np.asarray(poses[0]["world_T_camera"], dtype=float)
    processed = np.asarray(raw["processed_camera_intrinsics"], dtype=float)
    sx, sy = 439 / 640, 329 / 480
    full = processed.copy()
    full[0] = (processed[0] + np.array([0, 0, 60])) / sx
    full[1] = (processed[1] + np.array([0, 0, 4])) / sy
    if (
        not np.isfinite(transform).all()
        or transform.shape != (4, 4)
        or not np.allclose(full[:2, 2], [320, 240], atol=1e-7)
    ):
        raise ValueError("Ego seed camera/crop differs from the recorded 640 pipeline")
    points = []
    for candidate in raw["displayed_frontiers"]:
        camera = np.linalg.solve(transform, np.r_[candidate["world_xyz"], 1])[:3]
        if camera[2] <= 0:
            raise ValueError(
                "Recorded displayed frontier lies behind its capture camera"
            )
        expected = (full @ camera)[:2] / camera[2]
        pixel = (np.asarray(candidate["pixel_xy"]) + [60, 4]) / [sx, sy]
        if not np.allclose(expected, pixel, atol=1e-5):
            raise ValueError("World projection contradicts recorded FrontierNet pixel")
        if 0 <= pixel[0] < 640 and 0 <= pixel[1] < 480:
            points.append(
                dict(
                    label=candidate["label"],
                    frontier_id=candidate["frontier_id"],
                    xy=pixel.tolist(),
                    world_xyz=candidate["world_xyz"],
                )
            )
    return points, dict(
        candidate_path=str(path.resolve()),
        candidate_sha256=_sha(path),
        graph_path=str(graph_path.resolve()),
        graph_sha256=_sha(graph_path),
        observation_step=step,
        camera_pose="original exact observation graph world_T_camera",
        native_camera="frozen identity child of model camera, same copied optics",
        model_intrinsics_derived=full.tolist(),
        native_scale=4,
    )


def _ncc_match(template, image, point, radius=18):
    """Bounded normalized correlation; ambiguous/flat/edge matches fail closed."""
    from scipy.signal import fftconvolve

    half = template.shape[0] // 2
    x, y = np.rint(point).astype(int)
    left, top = x - half - radius, y - half - radius
    right, bottom = x + half + radius + 1, y + half + radius + 1
    if min(left, top) < 0 or right > image.shape[1] or bottom > image.shape[0]:
        return None
    region = image[top:bottom, left:right].astype(np.float64)
    centered = template.astype(np.float64) - float(template.mean())
    energy = float((centered**2).sum())
    if energy < template.size * 4:
        return None

    def window_sum(values):
        integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        n = template.shape[0]
        return (
            integral[n:, n:]
            - integral[:-n, n:]
            - integral[n:, :-n]
            + integral[:-n, :-n]
        )

    total = window_sum(region)
    square = window_sum(region * region)
    variance = np.maximum(square - total * total / template.size, 0)
    score = fftconvolve(region, centered[::-1, ::-1], mode="valid") / np.sqrt(
        np.maximum(variance * energy, 1e-12)
    )
    iy, ix = np.unravel_index(int(score.argmax()), score.shape)
    confidence = float(score[iy, ix])
    alternate = score.copy()
    alternate[max(0, iy - 3) : iy + 4, max(0, ix - 3) : ix + 4] = -1
    margin = confidence - float(alternate.max())
    if (
        confidence < 0.90
        or margin < 0.015
        or min(ix, iy, 2 * radius - ix, 2 * radius - iy) < 2
    ):
        return None
    return np.array([left + half + ix, top + half + iy], dtype=float), confidence


def track_point(previous, current, point):
    half = 16
    x, y = np.rint(point).astype(int)
    if (
        min(x - half, y - half) < 0
        or x + half >= previous.shape[1]
        or y + half >= previous.shape[0]
    ):
        return None
    template = previous[y - half : y + half + 1, x - half : x + half + 1]
    forward = _ncc_match(template, current, point)
    if forward is None:
        return None
    target, confidence = forward
    tx, ty = np.rint(target).astype(int)
    reverse = _ncc_match(
        current[ty - half : ty + half + 1, tx - half : tx + half + 1], previous, target
    )
    if reverse is None or np.linalg.norm(reverse[0] - [x, y]) > 1.5:
        return None
    return target, min(confidence, reverse[1])


def _badge_events(ass, frame, points, selected, native_size, display_affine=None):
    from .model_output_overlay import _frontier_badge_positions, _rounded_panel_path

    width, height = native_size
    mask = np.zeros((height, width), dtype=bool)
    q = height / 720
    for x, y, w, h in [(16, 16, 324, 132), (340, 16, 460, 132), (800, 16, 144, 132)]:
        mask[round(y * q) : round((y + h) * q), round(x * q) : round((x + w) * q)] = (
            True
        )
    # Reserve the central human-head area used by the reviewed initial views.
    # This is a layout exclusion, not a detector or an occlusion claim.
    mask[
        round(height * 0.15) : round(height * 0.41),
        round(width * 0.43) : round(width * 0.57),
    ] = True
    mask[round(height * 0.91) :, : round(width * 0.16)] = True
    mask[round(height * 0.91) :, round(width * 0.76) :] = True
    native_points = [(4 * p["xy"][0], 4 * p["xy"][1]) for p in points]
    if display_affine is not None:
        from .demo_stabilization import project_native_point

        native_points = [
            project_native_point(point, display_affine) for point in native_points
        ]
    start = math.floor(frame * 100 / 30) / 100
    end = math.floor((frame + 1) * 100 / 30) / 100
    visible = 0
    centres = []
    try:
        joint = _frontier_badge_positions(
            native_points, (16, 16, width - 16, height - 16), 75, mask
        )
    except ValueError:
        joint = [None] * len(points)
    # A crowded or protected point is hidden rather than covering scene/UI.
    for p, (x, y), centre in zip(points, native_points, joint):
        if not (0 <= x < width and 0 <= y < height):
            continue
        if centre is None:
            try:
                centre = _frontier_badge_positions(
                    [(x, y)], (16, 16, width - 16, height - 16), 75, mask
                )[0]
            except ValueError:
                continue
        if any(
            abs(centre[0] - cx) < 79 and abs(centre[1] - cy) < 79 for cx, cy in centres
        ):
            continue
        centres.append(centre)
        cx, cy = centre
        mask[
            max(0, round(cy - 41)) : round(cy + 41),
            max(0, round(cx - 41)) : round(cx + 41),
        ] = True
        active = p["label"] == selected
        fill, border, text = (
            ("7A516B", "ADDDC8", "FFFFFF") if active else ("FBF9FC", "7A516B", "7A516B")
        )
        dx, dy = cx - x, cy - y
        length = max(math.hypot(dx, dy), 1)
        nx, ny = -dy / length * 1.5, dx / length * 1.5
        vector = f"m {x + nx:.1f} {y + ny:.1f} l {cx + nx:.1f} {cy + ny:.1f} {cx - nx:.1f} {cy - ny:.1f} {x - nx:.1f} {y - ny:.1f}"
        ass.emit(start, end, r"{\an7\pos(0,0)\p1\bord0\shad0\1c&H7A516B&}" + vector, 10)
        for diameter, color in [(16, "FFFFFF"), (10, "7A516B")]:
            ass.emit(
                start,
                end,
                rf"{{\an7\pos({x - diameter / 2:.1f},{y - diameter / 2:.1f})\p1\bord0\shad0\1c&H{color}&}}"
                + _rounded_panel_path(diameter, diameter, diameter // 2),
                11,
            )
        ass.emit(
            start,
            end,
            rf"{{\an7\pos({cx - 37.5:.1f},{cy - 37.5:.1f})\p1\bord3\3c&H{border}&\shad0\1c&H{fill}&}}"
            + _rounded_panel_path(75, 75, 0),
            12,
        )
        ass.emit(
            start,
            end,
            rf"{{\an5\pos({cx:.1f},{cy:.1f})\fnTimes New Roman\fs51\b1\bord0\shad0\1c&H{text}&}}"
            + p["label"],
            13,
        )
        visible += 1
    return visible


def load_ego_frontier_cache(
    path, *, ego_path, source_steps, output_steps, model_source=None
):
    """Reuse reviewed displayed points, including their original hidden frames."""
    receipt = json.loads(Path(path).read_bytes())
    if receipt.get("schema_version") != "tarsphere.demo.ego_frontier_cache.v1":
        raise ValueError("Unsupported ego frontier cache receipt")
    for key in ("presentation", "ass"):
        if _sha(receipt[key]["path"]) != receipt[key]["sha256"]:
            raise ValueError("Ego frontier cache hash mismatch")
    metadata = json.loads(Path(receipt["presentation"]["path"]).read_bytes())
    report = metadata["ego_frontier_labels"]
    clock = hashlib.sha256(json.dumps(list(source_steps)).encode()).hexdigest()
    if (
        report["source_video_sha256"] != _sha(ego_path)
        or report["source_clock_sha256"] != clock
        or report["source_frames"] != len(source_steps)
        or metadata["frame_count"] != len(output_steps)
        or report["source_model_camera"] != model_source
        or Path(metadata["subtitle_file"]).resolve()
        != Path(receipt["ass"]["path"]).resolve()
    ):
        raise ValueError("Ego frontier cache differs from the original camera/clock")
    for source in report["source_candidates"]:
        for key in ("candidate", "graph"):
            if _sha(source[key + "_path"]) != source[key + "_sha256"]:
                raise ValueError("Ego frontier cache candidate source changed")
    if model_source:
        for key in ("ego", "frame_manifest"):
            if _sha(model_source[key]) != model_source[key + "_sha256"]:
                raise ValueError("Ego frontier cache model camera source changed")
    frames = {}

    def centiseconds(stamp):
        hours, minutes, seconds = stamp.split(":")
        return (int(hours) * 3600 + int(minutes) * 60) * 100 + round(
            float(seconds) * 100
        )

    for line in Path(receipt["ass"]["path"]).read_text().splitlines():
        if not line.startswith("Dialogue: "):
            continue
        fields = line.split(",", 9)
        layer = int(fields[0].split(":")[1])
        if layer not in (11, 13):
            continue
        start, end = map(centiseconds, fields[1:3])
        frame = math.ceil(start * 30 / 100)
        if (
            frame >= len(output_steps)
            or start != frame * 100 // 30
            or end != (frame + 1) * 100 // 30
        ):
            raise ValueError(
                "Cached badge does not bind exactly one original30fps frame"
            )
        event = fields[9]
        position = re.search(r"\\pos\(([-\d.]+),([-\d.]+)\)", event)
        if position is None:
            raise ValueError("Cached native badge lacks a point position")
        data = frames.setdefault(frame, dict(points=[], labels=[]))
        if layer == 11 and r"\1c&HFFFFFF&" in event:
            from .model_output_overlay import _rounded_panel_path

            if event.rsplit("}", 1)[-1] != _rounded_panel_path(16, 16, 8):
                raise ValueError("Cached point is not the original16px circle geometry")
            data["points"].append([float(position[1]) + 8, float(position[2]) + 8])
        elif layer == 13:
            label = event.rsplit("}", 1)[-1]
            if not re.fullmatch(r"[A-Z]", label):
                raise ValueError("Invalid cached frontier label")
            data["labels"].append(label)
    for frame, data in frames.items():
        if len(data["points"]) != len(data["labels"]) or len(
            set(data["labels"])
        ) != len(data["labels"]):
            raise ValueError("Cached native points and letters do not correspond")
        if any(
            not np.isfinite(p).all() or not (0 <= p[0] < 2560 and 0 <= p[1] < 1920)
            for p in data["points"]
        ):
            raise ValueError("Cached point lies outside native ego")
    if (
        sum(len(v["labels"]) for f, v in frames.items() if f < len(source_steps))
        != report["displayed_frame_labels"]
    ):
        raise ValueError("Cached point coverage differs from reviewed report")
    return metadata, frames


def append_cached_ego_frontier_labels(
    *,
    cache_path,
    ass_path,
    ego_path,
    source_steps,
    output_steps,
    trace,
    heartbeats,
    resolution,
    model_source=None,
    display_affines=None,
):
    from .model_output_overlay import (
        decision_indices_by_physics_step,
        _selected_frontier_label,
        _FocusAss,
    )

    metadata, frames = load_ego_frontier_cache(
        cache_path,
        ego_path=ego_path,
        source_steps=source_steps,
        output_steps=output_steps,
        model_source=model_source,
    )
    indices = decision_indices_by_physics_step(output_steps, trace, heartbeats)
    ass = _FocusAss(*resolution, font_family="Times New Roman")
    report = {
        **metadata["ego_frontier_labels"],
        "tracking_samples": [],
        "displayed_frame_labels": 0,
        "displayed_frames": 0,
        "editorial_hold_labels": 0,
    }
    report["display_coordinate_cache"] = {
        "path": str(Path(cache_path).resolve()),
        "sha256": _sha(cache_path),
        "mode": "reviewed ASS native16px point geometry; hidden frames remain hidden",
        "coordinate_rounding_max_px": 0.05,
        "ncc_recomputed": False,
    }
    if display_affines is not None and len(display_affines) != len(source_steps):
        raise ValueError("Display affine count differs from the original native clock")
    for frame, data in sorted(frames.items()):
        points = [
            dict(label=label, xy=np.asarray(point) / 4)
            for label, point in zip(data["labels"], data["points"])
        ]
        selected = _selected_frontier_label(trace[indices[frame]])
        matrix = (
            display_affines[min(frame, len(source_steps) - 1)]
            if display_affines is not None
            else None
        )
        shown = _badge_events(ass, frame, points, selected, (2560, 1920), matrix)
        if frame < len(source_steps):
            report["displayed_frame_labels"] += shown
            report["displayed_frames"] += bool(shown)
        else:
            report["editorial_hold_labels"] += shown
        if frame < len(source_steps) and frame % 15 == 0:
            if matrix is not None:
                from .demo_stabilization import project_native_point
            sample_points = []
            outside = []
            for p, xy in zip(points, data["points"]):
                native_xy = (
                    project_native_point(xy, matrix) if matrix is not None else list(xy)
                )
                sample = dict(
                    label=p["label"], source_native_xy=list(xy), native_xy=native_xy
                )
                (
                    sample_points
                    if 0 <= native_xy[0] < 2560 and 0 <= native_xy[1] < 1920
                    else outside
                ).append(sample)
            report["tracking_samples"].append(
                dict(
                    frame_index=frame,
                    physics_step=output_steps[frame],
                    selected=selected,
                    points=sample_points,
                    display_transform_out_of_view_hidden=outside,
                    displayed_labels=shown,
                )
            )
    with Path(ass_path).open("a") as handle:
        handle.write("\n".join(ass.events) + "\n")
    return report


def _captured_gray_frames(model_source, ticks, ffmpeg):
    """Read original 640 video at exact archived capture ticks; no nearest pose."""
    if model_source is None:
        return {}, {}
    for key in ("ego", "frame_manifest"):
        if _sha(model_source[key]) != model_source[key + "_sha256"]:
            raise ValueError("Original model camera source hash mismatch")
    rows = [
        json.loads(line)
        for line in Path(model_source["frame_manifest"]).read_text().splitlines()
        if line.strip()
    ]
    chosen = {}
    for row in rows:
        tick = int(row["physics_step"])
        if tick in ticks and tick not in chosen:
            chosen[tick] = int(row["frame_index"])
    indices = sorted(chosen.values())
    if not indices:
        return {}, {}
    expression = "+".join(f"eq(n,{i})" for i in indices)
    command = [
        ffmpeg,
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(model_source["ego"]),
        "-vf",
        f"select='{expression}',format=gray",
        "-fps_mode",
        "vfr",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    expected = len(indices) * 640 * 480
    if len(process.stdout) != expected:
        raise ValueError("Original model camera frames are missing or not 640x480")
    images = np.frombuffer(process.stdout, dtype=np.uint8).reshape(
        len(indices), 480, 640
    )
    by_index = dict(zip(indices, images))
    return {tick: by_index[index] for tick, index in chosen.items()}, chosen


def append_ego_frontier_labels(
    *,
    ass_path,
    ego_path,
    source_steps,
    output_steps,
    trace,
    heartbeats,
    online_map_dir,
    captures,
    resolution,
    ffmpeg="ffmpeg",
    model_source=None,
):
    from .model_output_overlay import (
        decision_indices_by_physics_step,
        online_map_paths_by_trace_index,
        _selected_frontier_label,
        _FocusAss,
    )

    map_paths = online_map_paths_by_trace_index(
        online_map_dir=online_map_dir, trace=trace
    )
    indices = decision_indices_by_physics_step(source_steps, trace, heartbeats)
    capture_by_observation = {int(c["observation_step"]): c for c in captures}
    seeds = {}
    for i, record in enumerate(trace):
        path = map_paths[i]
        step = int(path.name.split("frontier_step_")[1].split("_")[0])
        capture = capture_by_observation.get(step)
        if capture is None:
            continue
        candidate = path.with_name(
            path.name.replace("_exploration_memory.png", "_candidates.json")
        )
        points, provenance = captured_seed(candidate)
        seeds[i] = (int(capture["physics_step"]), points, {**provenance, **capture})
    model_images, model_indices = _captured_gray_frames(
        model_source, {s[0] for s in seeds.values()}, ffmpeg
    )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        str(ego_path),
        "-vf",
        "scale=640:480:flags=area,format=gray",
        "-frames:v",
        str(len(source_steps)),
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ass = _FocusAss(*resolution, font_family="Times New Roman")
    report = dict(
        mode="captured projection seeds plus image-space NCC tracking",
        measured_per_frame_camera_pose=False,
        working_resolution=[640, 480],
        native_font_px=51,
        thresholds=dict(ncc=0.90, ambiguity_margin=0.015, forward_backward_px=1.5),
        source_model_camera=model_source,
        image_anchored=0,
        source_video_sha256=_sha(ego_path),
        source_candidates=[],
        events=[],
        tracking_samples=[],
        seeded=0,
        matched=0,
        hidden=0,
        displayed_frame_labels=0,
        editorial_hold_labels=0,
    )
    unique_captures = {
        (seed[0], seed[2]["candidate_sha256"]) for seed in seeds.values()
    }
    report["captures_total"] = len(unique_captures)
    report["captures_coincident"] = sum(
        tick in source_steps for tick, _ in unique_captures
    )
    report["source_frames"] = len(source_steps)
    report["displayed_frames"] = 0
    previous = None
    active = []
    current_decision = None
    last_points = []
    try:
        for frame, (tick, index) in enumerate(zip(source_steps, indices)):
            data = process.stdout.read(640 * 480)
            if len(data) != 640 * 480:
                raise ValueError(
                    "Ego tracking decode ended before the original source clock"
                )
            gray = np.frombuffer(data, dtype=np.uint8).reshape(480, 640)
            if index != current_decision:
                # Reuse only the same captured observation on queued actions.
                old_seed = (
                    seeds.get(current_decision)
                    if current_decision is not None
                    else None
                )
                new_seed = seeds.get(index)
                if old_seed is None or new_seed is None or old_seed[0] != new_seed[0]:
                    active = []
                elif new_seed is not None:
                    active = [
                        p
                        for p in active
                        if any(
                            p["frontier_id"] == n["frontier_id"]
                            and p["label"] == n["label"]
                            and np.allclose(p["world_xyz"], n["world_xyz"])
                            for n in new_seed[1]
                        )
                    ]
                current_decision = index
            seed = seeds.get(index)
            if seed is not None and tick == seed[0]:
                active = [{**p, "xy": np.asarray(p["xy"])} for p in seed[1]]
                report["seeded"] += len(active)
                report["source_candidates"].append(
                    {**seed[2], "native_frame_index": frame}
                )
                report["events"].append(
                    dict(
                        frame_index=frame,
                        physics_step=tick,
                        event="exact_capture_anchor",
                        labels=[p["label"] for p in active],
                    )
                )
            elif seed is not None and not active and seed[0] <= tick <= seed[0] + 8:
                image = model_images.get(seed[0])
                if image is not None:
                    for point in seed[1]:
                        match = track_point(image, gray, point["xy"])
                        if match is not None:
                            active.append({**point, "xy": match[0]})
                    if active:
                        report["image_anchored"] += len(active)
                        report["source_candidates"].append(
                            {
                                **seed[2],
                                "native_frame_index": frame,
                                "native_frame_physics_step": tick,
                                "model_camera_frame_index": model_indices[seed[0]],
                                "anchor_method": "exact captured 640 image matched to subsequent native image",
                            }
                        )
                        report["events"].append(
                            dict(
                                frame_index=frame,
                                physics_step=tick,
                                source_capture_physics_step=seed[0],
                                event="captured_image_ncc_anchor",
                                labels=[p["label"] for p in active],
                            )
                        )
            elif previous is not None:
                tracked = []
                for point in active:
                    match = track_point(previous, gray, point["xy"])
                    if match is None:
                        report["hidden"] += 1
                        report["events"].append(
                            dict(
                                frame_index=frame,
                                physics_step=tick,
                                event="correspondence_failed_hide",
                                label=point["label"],
                            )
                        )
                    else:
                        xy, confidence = match
                        tracked.append({**point, "xy": xy})
                        report["matched"] += 1
                active = tracked
            selected = _selected_frontier_label(trace[index])
            shown = _badge_events(ass, frame, active, selected, (2560, 1920))
            report["displayed_frame_labels"] += shown
            report["displayed_frames"] += bool(shown)
            if (
                frame % 15 == 0
                or (seed is not None and tick == seed[0])
                or (
                    report["events"]
                    and report["events"][-1]["frame_index"] == frame
                    and report["events"][-1]["event"] == "captured_image_ncc_anchor"
                )
            ):
                report["tracking_samples"].append(
                    dict(
                        frame_index=frame,
                        physics_step=tick,
                        selected=selected,
                        points=[
                            dict(label=p["label"], native_xy=(4 * p["xy"]).tolist())
                            for p in active
                        ],
                        displayed_labels=shown,
                    )
                )
            previous = gray.copy()
            last_points = active
        if process.wait() != 0:
            raise RuntimeError(process.stderr.read().decode("utf8", "replace"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    for frame in range(len(source_steps), len(output_steps)):
        report["editorial_hold_labels"] += _badge_events(
            ass,
            frame,
            last_points,
            _selected_frontier_label(trace[indices[-1]]),
            (2560, 1920),
        )
    with Path(ass_path).open("a") as handle:
        handle.write("\n".join(ass.events) + "\n")
    report["source_clock_sha256"] = hashlib.sha256(
        json.dumps(list(source_steps)).encode()
    ).hexdigest()
    report["original_inputs_unchanged"] = True
    return report
