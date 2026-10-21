"""Render evaluated-model decisions as a policy-invisible ASS overlay.

The authoritative rollout trace supplies the text. Physics-step heartbeats
align every room-observer frame with the decision being executed, including
queued calls from a dynamic Action Horizon.
"""

from __future__ import annotations

import json
import re
import textwrap
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INTERACTION_MIN_DISPLAY_SECONDS = 3.0


@dataclass(frozen=True)
class _OverlayDecision:
    """Presentation-ready fields from one authoritative decision record."""

    policy_step: Any
    action_index: int
    sequence_length: int
    thinking: str
    actions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _OverlayUtterance:
    """One spoken utterance recorded by the authoritative rollout trace."""

    speaker_kind: str
    speaker_id: str
    message: str


@dataclass(frozen=True)
class _OverlayPlan:
    """Agent-authored ordered plan and its harness-tracked progress."""

    goal: str
    steps: tuple[str, ...]
    completed_steps: int | None
    terminal_confirmation_pending: bool = False


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(float(seconds) * 100.0))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _escape_ass(text: str) -> str:
    return (
        str(text)
        .replace("\\", "＼")
        .replace("{", "｛")
        .replace("}", "｝")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _wrapped_lines(text: str, *, width: int, prefix: str = "") -> list[str]:
    clean = " ".join(_escape_ass(text).split())
    if not clean:
        return [prefix.rstrip()]
    return textwrap.wrap(
        clean,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=True,
        break_on_hyphens=False,
    )


def _overlay_decision(decision: Mapping[str, Any]) -> _OverlayDecision | None:
    envelope = decision.get("envelope")
    if not isinstance(envelope, Mapping):
        return None
    thinking = str(envelope.get("thinking") or "").strip()
    raw_calls = envelope.get("actions")
    if not thinking or not isinstance(raw_calls, Sequence) or not raw_calls:
        return None

    calls: list[tuple[str, str]] = []
    for raw_call in raw_calls:
        call = raw_call if isinstance(raw_call, Mapping) else {}
        name = str(call.get("tool") or "unknown")
        args = call.get("args")
        if args is None:
            args = call.get("arguments")
        if isinstance(args, Mapping):
            args_text = (
                "  ·  ".join(
                    f"{key} = "
                    + json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(", ", ": "),
                        default=str,
                    )
                    for key, value in args.items()
                )
                or "no arguments"
            )
        else:
            args_text = json.dumps(
                args if args is not None else {},
                ensure_ascii=False,
                separators=(", ", ": "),
                default=str,
            )
        calls.append((name, args_text))
    return _OverlayDecision(
        policy_step=decision.get("policy_step"),
        action_index=int(decision.get("action_index") or 0),
        sequence_length=int(decision.get("action_sequence_length") or len(calls)),
        thinking=thinking,
        actions=tuple(calls),
    )


def _overlay_utterances(
    decision: Mapping[str, Any],
) -> tuple[_OverlayUtterance, ...]:
    """Extract spoken Robot/Human turns in their execution order."""

    utterances: list[_OverlayUtterance] = []
    robot_message = str(decision.get("robot_message") or "").strip()
    if robot_message:
        utterances.append(
            _OverlayUtterance(
                speaker_kind="ROBOT",
                speaker_id="Robot",
                message=robot_message,
            )
        )
    raw_events = decision.get("human_events")
    if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                continue
            if str(raw_event.get("action") or "").strip() != "speak":
                continue
            message = str(raw_event.get("message") or "").strip()
            if not message:
                continue
            human_id = str(raw_event.get("human_id") or "Human").strip()
            utterances.append(
                _OverlayUtterance(
                    speaker_kind="HUMAN",
                    speaker_id=human_id or "Human",
                    message=message,
                )
            )
    return tuple(utterances)


def _plan_update(decision: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the plan the Agent claims in its raw decision envelope."""

    envelope = decision.get("envelope")
    if not isinstance(envelope, Mapping):
        return None
    plan = envelope.get("plan")
    return plan if isinstance(plan, Mapping) else None


def _subgoal_checked(decision: Mapping[str, Any]) -> bool:
    """Read Agent-declared progress without exposing private backend state."""

    envelope = decision.get("envelope")
    return bool(isinstance(envelope, Mapping) and envelope.get("subgoal") == "checked")


def _plan_states_by_trace_index(
    trace: Sequence[Mapping[str, Any]],
) -> list[_OverlayPlan | None]:
    """Reconstruct only explicit Agent plan state; never infer completion."""

    current: _OverlayPlan | None = None
    states: list[_OverlayPlan | None] = []
    for decision in trace:
        update = _plan_update(decision)
        checked = _subgoal_checked(decision)
        if update is not None:
            goal = str(update.get("goal") or "").strip()
            raw_steps = update.get("steps")
            steps = tuple(
                item.strip()
                for item in (
                    raw_steps
                    if isinstance(raw_steps, Sequence)
                    and not isinstance(raw_steps, (str, bytes))
                    else ()
                )
                if isinstance(item, str) and item.strip()
            )
            current = _OverlayPlan(
                goal=goal,
                steps=steps,
                completed_steps=0,
            )
        if current is not None:
            completed = current.completed_steps or 0
            confirmation_pending = current.terminal_confirmation_pending
            if checked:
                final_step_index = len(current.steps) - 1
                if completed < final_step_index:
                    completed += 1
                    confirmation_pending = False
                elif completed == final_step_index:
                    if confirmation_pending:
                        completed += 1
                        confirmation_pending = False
                    else:
                        confirmation_pending = True
            elif confirmation_pending:
                confirmation_pending = False
            if (
                completed != current.completed_steps
                or confirmation_pending != current.terminal_confirmation_pending
            ):
                current = _OverlayPlan(
                    goal=current.goal,
                    steps=current.steps,
                    completed_steps=completed,
                    terminal_confirmation_pending=confirmation_pending,
                )
        states.append(current)
    return states


def format_model_output(decision: Mapping[str, Any]) -> str | None:
    """Format one exact reasoning/tool-call proposal for a top-right panel."""

    content = _overlay_decision(decision)
    if content is None:
        return None
    lines = [
        (
            f"MODEL OUTPUT  ·  POLICY {content.policy_step}  ·  "
            f"EXECUTING {content.action_index + 1}/{content.sequence_length}"
        ),
        "",
        "REASONING",
        *_wrapped_lines(content.thinking, width=58),
        "",
        "TOOL CALLS",
    ]
    for index, (name, args_text) in enumerate(content.actions):
        marker = "▶" if index == content.action_index else "•"
        lines.extend(
            _wrapped_lines(
                f"{name}({args_text})",
                width=58,
                prefix=f"{marker} {index + 1}. ",
            )
        )
    return "\n".join(lines)


def model_output_by_physics_step(
    physics_steps: Sequence[int],
    trace: Sequence[Mapping[str, Any]],
    heartbeats: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Resolve every frame to the decision whose action interval contains it."""

    if not physics_steps or not trace:
        return []
    decision_indices = decision_indices_by_physics_step(
        physics_steps,
        trace,
        heartbeats,
    )
    formatted = [format_model_output(decision) for decision in trace]
    if not any(formatted):
        return []

    outputs: list[str] = []
    last_text = next(text for text in formatted if text is not None)
    for decision_index in decision_indices:
        text = formatted[min(decision_index, len(formatted) - 1)]
        if text is not None:
            last_text = text
        outputs.append(last_text)
    return outputs


def decision_indices_by_physics_step(
    physics_steps: Sequence[int],
    trace: Sequence[Mapping[str, Any]],
    heartbeats: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Resolve frame steps to trace indices using action-end heartbeats."""

    endpoints: list[int] = []
    endpoint_indices: list[int] = []
    for index, heartbeat in enumerate(heartbeats[: len(trace)]):
        raw_step = heartbeat.get("physics_steps")
        if raw_step is None:
            continue
        step = int(raw_step)
        if endpoints and step < endpoints[-1]:
            raise ValueError("Heartbeat physics steps must be non-decreasing.")
        endpoints.append(step)
        endpoint_indices.append(index)
    indices: list[int] = []
    for raw_frame_step in physics_steps:
        frame_step = int(raw_frame_step)
        if endpoints:
            endpoint_offset = min(
                bisect_left(endpoints, frame_step), len(endpoints) - 1
            )
            decision_index = endpoint_indices[endpoint_offset]
        else:
            decision_index = 0
        indices.append(min(decision_index, len(trace) - 1))
    return indices


def _truncate_lines(lines: list[str], *, limit: int) -> list[str]:
    if limit < 1:
        return []
    if len(lines) <= limit:
        return lines
    clipped = lines[:limit]
    clipped[-1] = clipped[-1].rstrip() + "  …"
    return clipped


def _justify_lines(lines: list[str], *, width: int) -> list[str]:
    """Distribute word spacing so non-final prose lines use the full width."""

    justified: list[str] = []
    for index, line in enumerate(lines):
        if index + 1 == len(lines) or line.rstrip().endswith("…"):
            justified.append(line)
            continue
        words = line.split()
        gap_count = len(words) - 1
        missing = width - sum(len(word) for word in words) - gap_count
        if gap_count < 1 or missing <= 0:
            justified.append(line)
            continue
        base_extra, remainder = divmod(missing, gap_count)
        justified.append(
            "".join(
                word
                + (
                    " " * (1 + base_extra + (gap_index < remainder))
                    if gap_index < gap_count
                    else ""
                )
                for gap_index, word in enumerate(words)
            )
        )
    return justified


_ONLINE_MAP_STEP_RE = re.compile(
    r"frontier_step_(?P<step>\d+)_exploration_memory\.png$"
)


def online_map_paths_by_trace_index(
    *,
    online_map_dir: Path,
    trace: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Resolve each executed action to the newest map known at that action.

    OpenFrontier writes one model-visible exploration map before each action.
    Information-only correction turns can refresh the same action step, so the
    monotonically numbered observation directory is the authority for which
    snapshot is newest.  Missing later snapshots retain the previous map; a
    future snapshot is never pulled backwards into an earlier video frame.
    """

    root = Path(online_map_dir) / "open_frontier"
    candidates: dict[int, Path] = {}
    for path in sorted(
        root.glob("observation_*/frontier_step_*_exploration_memory.png")
    ):
        match = _ONLINE_MAP_STEP_RE.fullmatch(path.name)
        if match is not None:
            candidates[int(match.group("step"))] = path
    if not candidates:
        raise FileNotFoundError(
            f"No model-visible online-map snapshots found under {root}."
        )

    resolved: list[Path] = []
    current: Path | None = None
    for index, decision in enumerate(trace):
        raw_step = decision.get("action_step")
        action_step = index if raw_step is None else int(raw_step)
        if action_step in candidates:
            current = candidates[action_step]
        if current is None:
            raise RuntimeError(
                "The first executed action has no online-map snapshot; "
                "refusing to render a future map as if it were already known."
            )
        resolved.append(current)
    return resolved


def write_online_map_video(
    *,
    output_path: Path,
    online_map_dir: Path,
    physics_steps: Sequence[int],
    observer_fps: float,
    trace: Sequence[Mapping[str, Any]],
    heartbeats: Sequence[Mapping[str, Any]],
    resolution: tuple[int, int],
    paper: bool = False,
    native_frontier_badges: bool = True,
) -> Path:
    """Encode the online maps aligned to the continuous ego-video clock."""

    if observer_fps <= 0.0:
        raise ValueError("observer_fps must be positive")
    if not physics_steps or not trace:
        raise ValueError("online-map video requires frames and a decision trace")
    map_paths = online_map_paths_by_trace_index(
        online_map_dir=online_map_dir,
        trace=trace,
    )
    decision_indices = decision_indices_by_physics_step(
        physics_steps,
        trace,
        heartbeats,
    )

    from PIL import Image

    from tarsphere.infra.perception.continuous_ego_video import (
        _RawRgbVideoWriter,
    )

    output_path = Path(output_path)
    writer = _RawRgbVideoWriter(
        output_path=output_path,
        resolution=resolution,
        fps=observer_fps,
        **({"crf": 16} if paper else {}),
    )
    cached_path: Any = None
    cached_rgb: Any = None
    try:
        for decision_index in decision_indices:
            path = map_paths[min(decision_index, len(map_paths) - 1)]
            selected = (
                _selected_frontier_label(trace[decision_index])
                if paper and native_frontier_badges
                else None
            )
            cache_key = (path, selected)
            if cache_key != cached_path:
                with Image.open(path) as image:
                    frontiers = (
                        _recorded_map_frontiers(path, image)
                        if paper and native_frontier_badges
                        else None
                    )
                    cached_rgb = _render_online_map_panel(
                        image,
                        resolution,
                        paper=paper,
                        frontiers=frontiers,
                        selected_frontier=selected,
                    )
                    import numpy as np

                    cached_rgb = np.asarray(cached_rgb, dtype=np.uint8)
                cached_path = cache_key
            writer.write(cached_rgb)
    finally:
        writer.close()
    return output_path


def _selected_frontier_label(record):
    actions = (record.get("envelope") or {}).get("actions") or []
    if not actions:
        return None
    active = min(max(int(record.get("action_index") or 0), 0), len(actions) - 1)
    action = actions[active]
    return (
        (action.get("args") or {}).get("label")
        if action.get("tool") == "navigate_to_frontier"
        else None
    )


def _recorded_map_frontiers(path, image):
    """Project original world anchors using a PNG-verified memory-map crop.

    FrontierNet pixel_xy is deliberately unused. A different/stale occupancy
    snapshot or crop fails rather than displacing a point on the published map.
    """
    import math
    import numpy as np
    from PIL import Image
    from .online_mapping import OnlineOccupancyGrid
    from ..navigation.open_frontier.exploration_memory import (
        _square_crop,
        _online_frontier_mask,
    )

    path = Path(path)
    match = _ONLINE_MAP_STEP_RE.fullmatch(path.name)
    if match is None or image.width != image.height:
        raise ValueError(
            "Frontier labels require the original square memory-map snapshot"
        )
    step = int(match.group("step"))
    stem = path.name.removesuffix("_exploration_memory.png")
    candidates = json.loads(path.with_name(stem + "_candidates.json").read_bytes())
    graph = json.loads(path.with_name(stem + "_graph.json").read_bytes())
    if candidates.get("observation_step") != step:
        raise ValueError("Frontier candidate observation differs from the recorded map")
    rows = candidates["displayed_frontiers"]
    if not isinstance(rows, list) or len(rows) > 3:
        raise ValueError("Recorded map requires zero to three displayed frontiers")
    labels = [row["label"] for row in rows]
    if len(set(labels)) != len(labels) or any(
        label not in ("A", "B", "C") for label in labels
    ):
        raise ValueError("Invalid or duplicate recorded frontier label")
    if not rows:
        return []
    by_id = {row["frontier_id"]: row for row in graph["frontiers"]}
    points = []
    for row in rows:
        xyz = np.asarray(row["world_xyz"], dtype=float)
        if (
            xyz.shape != (3,)
            or not np.isfinite(xyz).all()
            or row["frontier_id"] not in by_id
            or by_id[row["frontier_id"]]["last_observation_step"] != step
            or not any(
                np.array_equal(xyz, observed["world_xyz"])
                and row["pixel_xy"] == observed["pixel_xy"]
                for observed in candidates["observed_frontiers"]
            )
        ):
            raise ValueError(
                "Displayed frontier does not match its recorded current observation"
            )
        points.append(xyz[:2])
    snapshot_path = path.parents[2] / f"online_map_observation_step_{step:04d}.npz"
    grid = OnlineOccupancyGrid.from_npz(snapshot_path)
    layers = grid.exploration_layers()
    history = grid.exploration_pose_history()
    if not history or not math.isfinite(grid.resolution):
        raise ValueError("Frontier map crop needs its original measured pose history")
    required = tuple(
        tuple(int(v) for v in grid.world_to_grid(np.asarray(p)))
        for p in (*history, *points)
    )
    row0, row1, col0, col1 = _square_crop(
        layers["observed"],
        required_cells=required,
        minimum_span_cells=math.ceil(8.0 / grid.resolution),
        margin_cells=math.ceil(1.0 / grid.resolution),
    )
    rgb = np.empty((*layers["observed"].shape, 3), dtype=np.uint8)
    rgb[:] = (205, 211, 221)
    rgb[layers["free"]] = (231, 246, 235)
    rgb[layers["occupied"]] = (42, 51, 68)
    rgb[_online_frontier_mask(observed=layers["observed"], free=layers["free"])] = (
        34,
        197,
        94,
    )
    reference = np.asarray(
        Image.fromarray(rgb[row0:row1, col0:col1][::-1, ::-1]).resize(
            image.size, Image.Resampling.NEAREST
        )
    )
    original = np.asarray(image.convert("RGB"))
    # Cone, route, badges and saved-entity overlays have distinct colours.
    known = np.zeros(original.shape[:2], dtype=bool)
    for color in ((205, 211, 221), (231, 246, 235), (42, 51, 68), (34, 197, 94)):
        known |= np.all(original == color, axis=-1)
    if known.sum() < 256 or not np.array_equal(original[known], reference[known]):
        raise ValueError(
            "Recorded memory-map pixels do not verify the occupancy crop; refusing frontier projection"
        )
    projected = []
    for row, xy in zip(rows, points):
        r, c = (int(v) for v in grid.world_to_grid(xy))
        point = (
            (col1 - 1 - c + 0.5) * image.width / (col1 - col0),
            (row1 - 1 - r + 0.5) * image.height / (row1 - row0),
        )
        projected.append(
            {
                **row,
                "memory_pixel_xy": tuple(
                    float(np.clip(v, 0, image.width - 1)) for v in point
                ),
                "memory_crop_rc": (row0, row1, col0, col1),
            }
        )
    return projected


def _frontier_badge_positions(points, bounds, diameter, protected_mask=None):
    """Move badges only; keep every leader endpoint at its measured point."""
    import math

    left, top, right, bottom = bounds
    half = diameter / 2
    placed = []
    for x, y in points:
        proposals = []
        for distance in (diameter, diameter * 1.6, diameter * 2.3, diameter * 3.1):
            for angle in (-45, 45, -135, 135, -90, 90, 0, 180):
                rad = math.radians(angle)
                cx = min(max(x + distance * math.cos(rad), left + half), right - half)
                cy = min(max(y + distance * math.sin(rad), top + half), bottom - half)
                if any(
                    abs(cx - px) < diameter + 4 and abs(cy - py) < diameter + 4
                    for px, py in placed
                ):
                    continue
                if any(
                    abs(cx - px) < half + 5 and abs(cy - py) < half + 5
                    for px, py in points
                ):
                    continue
                if protected_mask is not None:
                    import numpy as np

                    if protected_mask[
                        max(0, int(cy - half - 3)) : int(cy + half + 4),
                        max(0, int(cx - half - 3)) : int(cx + half + 4),
                    ].any():
                        continue
                    fraction = np.linspace(
                        min(1, 8 / max(1, math.hypot(cx - x, cy - y))),
                        1,
                        max(2, int(math.hypot(cx - x, cy - y))),
                    )
                    line_x = np.clip(
                        np.rint(x + (cx - x) * fraction).astype(int),
                        0,
                        protected_mask.shape[1] - 1,
                    )
                    line_y = np.clip(
                        np.rint(y + (cy - y) * fraction).astype(int),
                        0,
                        protected_mask.shape[0] - 1,
                    )
                    if protected_mask[line_y, line_x].any():
                        continue
                proposals.append(((cx - x) ** 2 + (cy - y) ** 2, cx, cy))
        if not proposals:
            raise ValueError(
                "Frontier badges cannot fit without overlap inside the map"
            )
        _, cx, cy = min(proposals)
        placed.append((cx, cy))
    return placed


def _style_online_map(image, *, paper=False):
    """Restyle known map palette entries without moving or adding any evidence."""
    import numpy as np
    from PIL import Image

    source = np.asarray(image.convert("RGB"))
    styled = source.copy()
    palette = {
        (205, 211, 221): (28, 30, 39),  # unknown area
        (231, 246, 235): (226, 239, 231),  # observed free area
        (42, 51, 68): (97, 105, 115),  # occupied cells
        (34, 197, 94): (92, 158, 141),  # observed boundaries
        (124, 58, 237): (187, 153, 211),  # semantic landmarks
        (37, 99, 235): (116, 84, 148),  # measured trail
        (14, 165, 233): (139, 185, 174),
        (245, 158, 11): (204, 146, 119),  # sensor view cone
    }
    if paper:
        palette = {
            (205, 211, 221): (243, 242, 246),
            (231, 246, 235): (213, 238, 231),
            (42, 51, 68): (131, 128, 143),
            (34, 197, 94): (72, 175, 163),
            (124, 58, 237): (133, 87, 160),
            (37, 99, 235): (107, 81, 122),
            (14, 165, 233): (82, 139, 188),
            (245, 158, 11): (216, 139, 89),
        }
    for original, color in palette.items():
        styled[np.all(source == original, axis=-1)] = color
    return Image.fromarray(styled)


def _render_online_map_panel(
    image, resolution, *, paper=False, frontiers=None, selected_frontier=None
):
    """A crisp map frame and semantic legend around the complete recorded map."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = resolution
    q = width / 288
    logical_height = height / q
    panel = Image.new("RGB", resolution, "#F9F7FB" if paper else "#15171E")
    draw = ImageDraw.Draw(panel)

    def box(rect, **kwargs):
        if paper:
            kwargs["radius"] = 0
        draw.rounded_rectangle(tuple(round(v * q) for v in rect), **kwargs)

    def font(size, bold=False):
        names = [
            (
                (
                    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
                    if bold
                    else "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
                )
                if paper
                else (
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                    if bold
                    else "/System/Library/Fonts/Supplemental/Arial.ttf"
                )
            ),
            (
                (
                    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
                    if bold
                    else "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
                )
                if paper
                else (
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                    if bold
                    else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                )
            ),
        ]
        for path in names:
            if Path(path).is_file():
                return ImageFont.truetype(path, round(size * q))
        if paper:
            raise FileNotFoundError(
                "Times New Roman is required for the paper map frame"
            )
        return ImageFont.load_default(size=round(size * q))

    box(
        (0, 0, 287, logical_height - 1),
        radius=round(12 * q),
        fill="#FFFFFF" if paper else "#1C1E27",
        outline="#CCA2BE" if paper else "#464052",
        width=max(1, round(q)),
    )
    # Source PNG stays complete. Only its presentation frame/legend are typeset.
    if paper:
        box((0, 0, 288, 36), radius=0, fill="#EBDCF1")
        draw.text(
            (0, 11 * q), "SEMANTIC SPATIAL MEMORY", font=font(12, True), fill="#6B517A"
        )
        draw.text((268 * q, 13 * q), "2D", font=font(9), fill="#64616A")
        draw.line(
            [(0, 36 * q), (288 * q, 36 * q)], fill="#CCA2BE", width=max(1, round(q))
        )
    else:
        box((13, 12, 19, 18), radius=max(1, round(q)), fill="#9CD6BC")
        draw.text(
            (26 * q, 9 * q),
            "SEMANTIC SPATIAL MEMORY",
            font=font(9, True),
            fill="#E9E5EF",
        )
        draw.text((259 * q, 10 * q), "2D", font=font(9), fill="#9E9CAA")
        draw.line(
            [(12 * q, 28 * q), (275 * q, 28 * q)],
            fill="#393543",
            width=max(1, round(q)),
        )
    map_y = 44 if paper else 32
    logical_map_size = min(234, logical_height - (66 if paper else 54))
    map_size = round(logical_map_size * q)
    recorded = _style_online_map(image, paper=paper).resize(
        (map_size, map_size), Image.Resampling.LANCZOS
    )
    panel.paste(recorded, (round((288 - logical_map_size) / 2 * q), round(map_y * q)))
    draw = ImageDraw.Draw(panel)
    if frontiers:
        import numpy as np

        # Native panel pixels: badges are never drawn into the model's 320px PNG.
        mx, my = round((288 - logical_map_size) / 2 * q), round(map_y * q)
        points = [
            (
                mx + f["memory_pixel_xy"][0] * map_size / image.width,
                my + f["memory_pixel_xy"][1] * map_size / image.height,
            )
            for f in frontiers
        ]
        diameter = round(28 * q)
        source = np.asarray(image.convert("RGB"))
        robot = (
            (source[:, :, 0] > 160) & (source[:, :, 1] < 75) & (source[:, :, 2] < 120)
        )
        entities = (
            (source[:, :, 0] > 90)
            & (source[:, :, 0] < 160)
            & (source[:, :, 1] < 105)
            & (source[:, :, 2] > 170)
        )
        protected = np.zeros((height, width), dtype=bool)
        protected[my : my + map_size, mx : mx + map_size] = (
            np.asarray(
                Image.fromarray((robot | entities).astype(np.uint8) * 255).resize(
                    (map_size, map_size), Image.Resampling.NEAREST
                )
            )
            > 0
        )
        badges = _frontier_badge_positions(
            points,
            (mx + 4 * q, my + 4 * q, mx + map_size - 4 * q, my + map_size - 4 * q),
            diameter,
            protected_mask=protected,
        )
        for f, point, badge in zip(frontiers, points, badges):
            selected = f["label"] == selected_frontier
            color = "#6B517A" if paper else "#BB99D3"
            draw.line([point, badge], fill="#FFFFFF", width=max(3, round(4 * q)))
            draw.line([point, badge], fill=color, width=max(1, round(1.2 * q)))
            radius = 3 * q
            draw.ellipse(
                (
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                ),
                fill="#6B517A" if selected else "#FFFFFF",
                outline=color,
                width=max(1, round(q)),
            )
        for f, (cx, cy) in zip(frontiers, badges):
            selected = f["label"] == selected_frontier
            half = diameter / 2
            draw.rounded_rectangle(
                (cx - half, cy - half, cx + half, cy + half),
                radius=7 * q,
                fill="#6B517A" if selected else "#FFFFFF",
                outline="#4AAF9F" if selected else "#9A7CAB",
                width=max(2, round((2 if selected else 1.2) * q)),
            )
            letter_font = font(19, True)
            b = draw.textbbox((0, 0), f["label"], font=letter_font)
            draw.text(
                (cx - (b[2] - b[0]) / 2 - b[0], cy - (b[3] - b[1]) / 2 - b[1]),
                f["label"],
                font=letter_font,
                fill="#FFFFFF" if selected else "#6B517A",
            )
    # Colors and labels follow exploration_memory.py: free, trail, saved entities.
    legend_y = logical_height - 16
    draw.rectangle(
        (14 * q, (legend_y + 3) * q, 20 * q, (legend_y + 9) * q),
        fill="#D5EEE7" if paper else "#E2EFE7",
    )
    draw.text(
        (25 * q, legend_y * q),
        "Free",
        font=font(9),
        fill="#64616A" if paper else "#B9B4C0",
    )
    draw.line(
        [(94 * q, (legend_y + 6) * q), (105 * q, (legend_y + 6) * q)],
        fill="#6B517A" if paper else "#745494",
        width=max(2, round(2 * q)),
    )
    draw.text(
        (111 * q, legend_y * q),
        "Trail",
        font=font(9),
        fill="#64616A" if paper else "#B9B4C0",
    )
    draw.ellipse(
        (185 * q, (legend_y + 3) * q, 191 * q, (legend_y + 9) * q),
        fill="#8557A0" if paper else "#BB99D3",
    )
    draw.text(
        (197 * q, legend_y * q),
        "Anchor",
        font=font(9),
        fill="#64616A" if paper else "#B9B4C0",
    )
    return panel


def _rounded_panel_path(width: int, height: int, radius: int) -> str:
    """Return an ASS vector path for a rounded top-right card."""

    w = int(width)
    h = int(height)
    r = max(1, min(int(radius), w // 2, h // 2))
    return (
        f"m {r} 0 l {w - r} 0 "
        f"b {w - r // 2} 0 {w} {r // 2} {w} {r} "
        f"l {w} {h - r} "
        f"b {w} {h - r // 2} {w - r // 2} {h} {w - r} {h} "
        f"l {r} {h} "
        f"b {r // 2} {h} 0 {h - r // 2} 0 {h - r} "
        f"l 0 {r} b 0 {r // 2} {r // 2} 0 {r} 0"
    )


# Display colors brighten the paper's purple/teal accents for dark video surfaces.
_HUD_COLORS = {
    "ink": "201B19",
    "surface": "292421",
    "line": "51493E",
    "white": "F6F3F0",
    "muted": "B8B3AE",
    "purple": "D8AFCA",
    "teal": "BCD69C",
    "paper": "F7F5F2",
    "coral": "889CE8",
}


class _FocusAss:
    """Pixel-measured typography, bounded text, and short event transitions."""

    def __init__(self, width, height, *, font_family=None):
        from PIL import ImageFont

        self.q = height / 720
        self.ImageFont = ImageFont
        self.font_file = None
        candidates = (
            [
                (
                    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                    "Times New Roman",
                ),
                (
                    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
                    "Times New Roman",
                ),
            ]
            if font_family == "Times New Roman"
            else [
                ("/System/Library/Fonts/Supplemental/Arial.ttf", "Arial"),
                ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVu Sans"),
            ]
        )
        for path, name in candidates:
            if Path(path).is_file():
                self.font_file, self.font_name = path, name
                break
        if self.font_file is None:
            if font_family == "Times New Roman":
                raise FileNotFoundError(
                    "Times New Roman is required for the paper video HUD"
                )
            self.font_name = "DejaVu Sans"
        self._font_cache = {}
        self.events = []
        self.header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Focus,{self.font_name},24,&H00F6F3F0,&H00FFFFFF,&H90000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def emit(self, start, end, text, layer=4):
        if end > start:
            self.events.append(
                f"Dialogue: {layer},{_ass_timestamp(start)},"
                f"{_ass_timestamp(end)},Focus,,0,0,0,,{text}"
            )

    def position(self, x, y, animate=False):
        x, y = round(x * self.q), round(y * self.q)
        if animate:
            return rf"\move({x},{y + round(6 * self.q)},{x},{y},0,180)\fad(160,120)"
        return rf"\pos({x},{y})"

    def text(
        self,
        start,
        end,
        x,
        y,
        text,
        *,
        size=17,
        color="white",
        bold=False,
        animate=False,
        layer=4,
    ):
        self.emit(
            start,
            end,
            r"{\an7"
            + self.position(x, y, animate)
            + rf"\fs{size * self.q:.2f}\c&H{_HUD_COLORS[color]}&\b{int(bold)}\q2}}"
            + _escape_ass(text),
            layer,
        )

    def font(self, size, bold=False):
        px = max(8, round(size * self.q))
        key = (px, bold)
        if key not in self._font_cache:
            path = self.font_file
            if path and bold:
                candidate = Path(path).with_name(
                    {
                        "Arial": "Arial Bold.ttf",
                        "Times New Roman": "Times New Roman Bold.ttf",
                    }.get(self.font_name, "DejaVuSans-Bold.ttf")
                )
                if candidate.is_file():
                    path = str(candidate)
            self._font_cache[key] = (
                self.ImageFont.truetype(path, px)
                if path
                else self.ImageFont.load_default(size=px)
            )
        return self._font_cache[key]

    def wrap(self, text, width, size, max_lines, bold=False):
        font = self.font(size, bold)
        limit = width * self.q * 0.98
        words = _escape_ass(text).split()
        lines = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if font.getlength(candidate) <= limit:
                current = candidate
            else:
                if current:
                    lines.append(current)
                # Handle long IDs without drawing outside the reserved width.
                while font.getlength(word) > limit and len(word) > 1:
                    n = len(word) - 1
                    while n > 1 and font.getlength(word[:n]) > limit:
                        n -= 1
                    lines.append(word[:n])
                    word = word[n:]
                current = word
        if current:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            tail = lines[-1]
            while tail and font.getlength(tail + "…") > limit:
                tail = tail[:-1]
            lines[-1] = tail.rstrip() + "…"
        return lines

    def paragraph(
        self,
        start,
        end,
        x,
        y,
        text,
        width,
        *,
        size=17,
        max_lines=3,
        color="white",
        bold=False,
        animate=False,
    ):
        lines = self.wrap(text, width, size, max_lines, bold)
        for i, line in enumerate(lines):
            self.text(
                start,
                end,
                x,
                y + i * size * 1.25,
                line,
                size=size,
                color=color,
                bold=bold,
                animate=animate,
            )
        return len(lines) * size * 1.25

    def box(
        self,
        start,
        end,
        x,
        y,
        w,
        h,
        *,
        color="ink",
        alpha="20",
        radius=10,
        animate=False,
        layer=1,
    ):
        path = _rounded_panel_path(
            round(w * self.q), round(h * self.q), round(radius * self.q)
        )
        self.emit(
            start,
            end,
            r"{\an7"
            + self.position(x, y, animate)
            + rf"\p1\bord0\shad0\1c&H{_HUD_COLORS[color]}&\1a&H{alpha}&}}"
            + path,
            layer,
        )

    def rule(self, start, end, x, y, w, *, color="line", thickness=1, layer=2):
        self.box(
            start,
            end,
            x,
            y,
            w,
            thickness,
            color=color,
            alpha="00",
            radius=0.5,
            layer=layer,
        )


def _friendly_action(name):
    return {
        "move_body": "Move",
        "navigate_to_entity": "Navigate",
        "navigate_to_pointgoal": "Navigate",
        "navigate_to_landmark": "Navigate",
        "explore_frontier": "Explore",
        "ask": "Ask",
        "speak": "Speak",
        "look_around": "Look around",
        "grasp_object": "Grasp",
        "place_object": "Place",
        "wait": "Wait",
        "remember_entity": "Save a visual anchor",
    }.get(name, name.replace("_", " ").capitalize())


def _compact_arguments(text):
    # Presentation rounding only; authoritative trace values remain unchanged.
    return re.sub(
        r"(?<![\w.])-?\d+\.\d{3,}", lambda m: f"{float(m.group()):.2f}", str(text)
    )


def _action_caption(tool, args):
    """Short labels from exact tool arguments; never infer a destination."""
    label = _friendly_action(tool)
    detail = _compact_arguments(args)
    if tool == "move_body":
        match = re.search(r"delta\s*=\s*(\[[^\]]+\])", args)
        if match:
            try:
                forward, lateral, turn = map(float, json.loads(match.group(1)))
                pieces = []
                if abs(forward) > 0.001:
                    label = "Move forward" if forward > 0 else "Move backward"
                    pieces.append(f"{abs(forward):g} m")
                if abs(lateral) > 0.001:
                    pieces.append(f"{abs(lateral):g} m sideways")
                if abs(turn) > 0.001:
                    if not pieces:
                        label = "Turn"
                    pieces.append(
                        f"{abs(turn):g}° " + ("left" if turn > 0 else "right")
                    )
                if pieces:
                    detail = " · ".join(pieces)
            except (ValueError, TypeError):
                pass
    else:
        for key in ["recipient", "entity_id", "object_id"]:
            match = re.search(r"(?<!\w)" + key + r'\s*=\s*"([^"\n]+)"', args)
            if match:
                target = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", match.group(1)).replace(
                    "_", " "
                )
                detail = ("To " if tool in ["ask", "speak"] else "Target: ") + target
                break
    return label, detail


def _current_task_text(content, plan, legacy=False):
    """Use a declared current subgoal, otherwise an exact excerpt of recorded intent."""
    if plan and plan.steps and not legacy and plan.completed_steps is not None:
        return plan.steps[min(plan.completed_steps, len(plan.steps) - 1)]
    text = content.thinking.strip()
    if text.lower().startswith("plan:"):
        text = text[5:].strip()
        text = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text, maxsplit=1)[0]
        match = re.search(
            r"(?:the )?(?:next unfinished step is|next objective (?:remains|is)|remaining step is)\s+(.+)",
            text,
            re.I,
        )
        if match:
            text = match.group(1)
        elif text.lower().startswith("first ") and ", then " in text:
            text = text.split(", then ", 1)[0]
        return text[:1].upper() + text[1:]
    if plan and plan.goal:
        return plan.goal
    return _friendly_action(
        content.actions[min(content.action_index, len(content.actions) - 1)][0]
    )


def write_model_output_ass(
    *,
    output_path: Path,
    physics_steps: Sequence[int],
    observer_fps: float,
    trace: Sequence[Mapping[str, Any]],
    heartbeats: Sequence[Mapping[str, Any]],
    resolution: tuple[int, int],
    panel_region=None,
    interaction_region=None,
    plan_region=None,
    task_title: str | None = None,
    source_label: str | None = None,
    lidar_by_trace=None,
) -> Path | None:
    """Keep the original ego/map layout; emphasize current intent and real events.

    The task title is optional editorial copy. Everything else is taken from
    the trace. No completion or remembered fact is inferred from evaluator data.
    """
    if observer_fps <= 0:
        raise ValueError("observer_fps must be positive")
    if not physics_steps or not trace:
        return None
    decisions = [_overlay_decision(t) for t in trace]
    if not any(decisions):
        return None
    width, height = map(int, resolution)
    if min(width, height) <= 0:
        raise ValueError("resolution must be positive")
    q = height / 720
    panel_region = panel_region or (
        round(width * 0.65),
        16,
        round(width * 0.33),
        height - 32,
    )
    for name, rect in [
        ("panel", panel_region),
        ("plan", plan_region),
        ("interaction", interaction_region),
    ]:
        if rect is None:
            continue
        x, y, w, h = rect
        if min(w, h) < 48 * q or min(x, y) < 0 or x + w > width or y + h > height:
            raise ValueError(f"{name}_region must be a usable in-frame rectangle")
    indices = decision_indices_by_physics_step(physics_steps, trace, heartbeats)
    plans = _plan_states_by_trace_index(trace)
    ass = _FocusAss(width, height)
    end_time = len(physics_steps) / observer_fps + INTERACTION_MIN_DISPLAY_SECONDS
    px, py, pw, ph = (v / q for v in panel_region)
    sx, sw = px + 16, pw - 32
    bottom = (py + ph) - 16
    # The sidebar is one quiet surface. Original logo is composited in its footer.
    ass.rule(0, end_time, sx, py + 5, sw, color="purple", thickness=2)
    if source_label:
        ass.paragraph(
            0,
            end_time,
            sx,
            bottom - 19,
            source_label,
            sw,
            size=10.5,
            max_lines=1,
            color="muted",
        )
    else:
        ass.text(
            0,
            end_time,
            sx,
            bottom - 19,
            "RECORDED AGENT VIEW",
            size=10.5,
            color="muted",
        )
    # Keep action and reasoning synchronized to their executed physics intervals.
    last = None
    frame_states = []
    for index in indices:
        last = decisions[index] or last
        frame_states.append(
            (last, plans[index], bool(trace[index].get("presentation_legacy_schema")))
        )
    start = 0
    while start < len(frame_states):
        stop = start + 1
        while stop < len(frame_states) and frame_states[stop] == frame_states[start]:
            stop += 1
        begin = start / observer_fps
        end = end_time if stop == len(frame_states) else stop / observer_fps
        content, plan, legacy = frame_states[start]
        if content is None:
            start = stop
            continue
        active = min(max(content.action_index, 0), len(content.actions) - 1)
        tool, args = content.actions[active]
        ass.text(begin, end, sx, py + 24, "NOW", size=10.5, color="purple", bold=True)
        ass.text(
            begin,
            end,
            sx + sw - 90,
            py + 24,
            f"ACTION {active + 1:02d}/{max(content.sequence_length, len(content.actions)):02d}",
            size=10,
            color="muted",
        )
        action_label, action_detail = _action_caption(tool, args)
        used = ass.paragraph(
            begin, end, sx, py + 45, action_label, sw, size=25, max_lines=2, bold=True
        )
        next_y = py + 53 + used
        ass.paragraph(
            begin,
            end,
            sx,
            next_y,
            action_detail,
            sw,
            size=12,
            max_lines=2,
            color="muted",
        )
        reasoning_y = max(py + 122, next_y + 32)
        ass.rule(begin, end, sx, reasoning_y - 15, sw)
        ass.text(
            begin,
            end,
            sx,
            reasoning_y,
            "REASONING",
            size=10.5,
            color="purple",
            bold=True,
        )
        reason_size = 14.5
        reason_lines = max(
            3, min(8, int((bottom - 90 - reasoning_y - 21) / (reason_size * 1.25)))
        )
        ass.paragraph(
            begin,
            end,
            sx,
            reasoning_y + 21,
            content.thinking,
            sw,
            size=reason_size,
            max_lines=reason_lines,
        )
        start = stop
    # Total goal and current task are separate: intent never comes from evaluator state.
    if plan_region:
        ex, ey, ew, eh = (v / q for v in plan_region)
        gx, gy, gw = ex + 16, ey + 16, ew - 272
        task_states = []
        last = None
        initial_goal = None
        for index in indices:
            last = decisions[index] or last
            plan = plans[index]
            goal = plan.goal if plan and plan.goal else (task_title or "")
            if initial_goal is None and goal:
                initial_goal = goal
            display_goal = task_title if task_title and goal == initial_goal else goal
            legacy = bool(trace[index].get("presentation_legacy_schema"))
            current = _current_task_text(last, plan, legacy) if last else ""
            task_states.append((display_goal, current, goal))
        start = 0
        previous_goal = ""
        while start < len(task_states):
            stop = start + 1
            while stop < len(task_states) and task_states[stop] == task_states[start]:
                stop += 1
            display_goal, current, goal = task_states[start]
            if display_goal or current:
                begin = start / observer_fps
                end = end_time if stop == len(task_states) else stop / observer_fps
                changed = bool(previous_goal and goal != previous_goal)
                lines = ass.wrap(current, gw - 38, 19, 2, True)
                gh = 150
                ass.box(begin, end, gx, gy, gw, gh, alpha="42", radius=10)
                ass.text(
                    begin,
                    end,
                    gx + 19,
                    gy + 14,
                    "GOAL",
                    size=10.5,
                    color="teal",
                    bold=True,
                )
                ass.paragraph(
                    begin,
                    end,
                    gx + 65,
                    gy + 12,
                    display_goal,
                    gw - 84,
                    size=13.5,
                    max_lines=1,
                    color="white",
                )
                ass.rule(begin, end, gx + 19, gy + 40, gw - 38)
                # A compact play/task glyph denotes the active task.
                ass.box(
                    begin,
                    end,
                    gx + 19,
                    gy + 52,
                    19,
                    19,
                    color="purple",
                    alpha="18",
                    radius=5,
                    layer=2,
                )
                ass.text(
                    begin, end, gx + 24, gy + 52, "▸", size=16, color="ink", bold=True
                )
                ass.text(
                    begin,
                    end,
                    gx + 47,
                    gy + 56,
                    "CURRENT TASK",
                    size=10.5,
                    color="muted",
                    bold=True,
                )
                ass.paragraph(
                    begin,
                    end,
                    gx + 19,
                    gy + 81,
                    current,
                    gw - 38,
                    size=19,
                    max_lines=2,
                    bold=True,
                )
                if changed:
                    flash_end = min(end, begin + 3.5)
                    ass.rule(begin, flash_end, gx, gy, gw, color="teal", thickness=3)
                    ass.text(
                        begin,
                        flash_end,
                        gx + gw - 120,
                        gy + 56,
                        "GOAL UPDATED",
                        size=10.5,
                        color="teal",
                        bold=True,
                    )
                previous_goal = goal
            start = stop
    # Speech is the visual highlight, appearing only during recorded interactions.
    if interaction_region:
        ex, ey, ew, eh = (v / q for v in interaction_region)
        events = []
        for index, decision in enumerate(trace):
            utterances = _overlay_utterances(decision)
            frames = [i for i, j in enumerate(indices) if j == index]
            if utterances and frames:
                begin = frames[0] / observer_fps
                natural = (frames[-1] + 1) / observer_fps
                events.append(
                    (
                        begin,
                        max(natural, begin + INTERACTION_MIN_DISPLAY_SECONDS),
                        utterances,
                    )
                )
        for number, (begin, natural, utterances) in enumerate(events):
            end = min(
                natural, events[number + 1][0] if number + 1 < len(events) else end_time
            )
            cw = ew - 32
            parts = []
            for u in utterances[-2:]:
                name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", u.speaker_id).replace(
                    "_", " "
                )
                lines = ass.wrap(u.message, cw - 48, 22, 3)
                parts.append((u, name, lines))
            ch = 28 + sum(24 + len(lines) * 27.5 + 14 for _, _, lines in parts)
            cx, cy = ex + 16, ey + eh - ch - 16
            ass.box(begin, end, cx, cy, cw, ch, alpha="22", radius=12, animate=True)
            row = cy + 17
            for u, name, lines in parts:
                accent = "teal" if u.speaker_kind == "HUMAN" else "purple"
                ass.text(
                    begin,
                    end,
                    cx + 24,
                    row,
                    name.upper(),
                    size=11,
                    color=accent,
                    bold=True,
                    animate=True,
                )
                row += 24
                for line_index, line in enumerate(lines):
                    ass.text(begin, end, cx + 24, row, line, size=22, animate=True)
                    row += 27.5
                row += 14
            ass.box(
                begin,
                end,
                cx,
                cy + 16,
                3,
                ch - 32,
                color="teal",
                alpha="00",
                radius=1,
                animate=True,
                layer=2,
            )
        # Show explicit write instructions as events, not as a verified ledger.
        # Initial setup notes stay quiet while the task is introduced.
        memory_events = []
        for index, decision in enumerate(trace):
            if index == 0:
                continue
            frames = [i for i, j in enumerate(indices) if j == index]
            if not frames:
                continue
            envelope = decision.get("envelope")
            if not isinstance(envelope, Mapping):
                continue
            changes = envelope.get("memory", [])
            if not isinstance(changes, Sequence) or not changes:
                continue
            change = changes[-1]
            if not isinstance(change, Mapping):
                continue
            op = str(change.get("op", "")).lower()
            label = {
                "remember": "MEMORY WRITE",
                "revise": "MEMORY REVISION",
                "forget": "MEMORY REMOVAL",
            }.get(op)
            content = str(change.get("note") or change.get("content") or "")
            if not label or not content:
                continue
            begin = frames[0] / observer_fps
            natural = max((frames[-1] + 1) / observer_fps, begin + 4)
            memory_events.append((begin, natural, label, content))
        for n, (begin, natural, label, content) in enumerate(memory_events):
            end = min(
                end_time,
                natural,
                memory_events[n + 1][0] if n + 1 < len(memory_events) else end_time,
            )
            mx, my, mw = ex + 16, ey + 178, ew - 272
            mh = 86
            ass.box(begin, end, mx, my, mw, mh, alpha="28", radius=10, animate=True)
            ass.text(
                begin,
                end,
                mx + 18,
                my + 13,
                label,
                size=10.5,
                color="teal",
                bold=True,
                animate=True,
            )
            ass.paragraph(
                begin,
                end,
                mx + 18,
                my + 35,
                content,
                mw - 36,
                size=16,
                max_lines=2,
                animate=True,
            )
        # Outcomes enter the view only when the action-end heartbeat is reached.
        from tarsphere.infra.perception.presentation_radar import (
            execution_feedback_caption,
        )

        feedback_events = []
        for index, heartbeat in enumerate(heartbeats[: len(trace)]):
            step = heartbeat.get("physics_steps")
            if step is None:
                continue
            frame = bisect_left(physics_steps, int(step))
            if frame >= len(physics_steps):
                continue
            caption = execution_feedback_caption(trace[index])
            if caption:
                feedback_events.append((frame / observer_fps, caption))
        memory_windows = [
            (
                begin,
                min(
                    natural,
                    end_time,
                    memory_events[i + 1][0] if i + 1 < len(memory_events) else end_time,
                ),
            )
            for i, (begin, natural, _, _) in enumerate(memory_events)
        ]
        for i, (begin, (label, detail, color)) in enumerate(feedback_events):
            end = min(
                begin + 4,
                end_time,
                feedback_events[i + 1][0] if i + 1 < len(feedback_events) else end_time,
            )
            spans = [(begin, end)]
            for mb, me in memory_windows:
                spans = [
                    piece
                    for a, b in spans
                    for piece in (
                        [(a, b)]
                        if me <= a or mb >= b
                        else [(a, min(b, mb)), (max(a, me), b)]
                    )
                    if piece[1] > piece[0]
                ]
            for a, b in spans:
                fx, fy, fw = ex + 16, ey + 178, ew - 272
                fh = 86
                ass.box(a, b, fx, fy, fw, fh, alpha="35", radius=10)
                ass.text(a, b, fx + 18, fy + 11, label, size=10, color=color, bold=True)
                ass.paragraph(
                    a, b, fx + 18, fy + 31, detail, fw - 36, size=15, max_lines=2
                )
        if lidar_by_trace and any(lidar_by_trace):
            from tarsphere.infra.perception.presentation_radar import draw_radar

            scans = []
            last_scan = None
            for index in indices:
                if index < len(lidar_by_trace) and lidar_by_trace[index] is not None:
                    last_scan = lidar_by_trace[index]
                scans.append(last_scan)
            start = 0
            while start < len(scans):
                stop = start + 1
                while stop < len(scans) and scans[stop] == scans[start]:
                    stop += 1
                if scans[start] is not None:
                    draw_radar(
                        ass,
                        start / observer_fps,
                        end_time if stop == len(scans) else stop / observer_fps,
                        ex + ew - 244,
                        ey + 16,
                        scans[start],
                    )
                start = stop
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass.header + "\n".join(ass.events) + "\n", encoding="utf-8")
    return output_path


def ffmpeg_subtitles_filter(path: Path) -> str:
    """Return a libavfilter-safe subtitles expression for an ASS sidecar."""

    escaped = str(path.resolve()).replace("\\", r"\\").replace(":", r"\:")
    escaped = escaped.replace("'", r"\'").replace(",", r"\,")
    escaped = escaped.replace("[", r"\[").replace("]", r"\]")
    return f"subtitles=filename='{escaped}'"


__all__ = [
    "INTERACTION_MIN_DISPLAY_SECONDS",
    "decision_indices_by_physics_step",
    "ffmpeg_subtitles_filter",
    "format_model_output",
    "model_output_by_physics_step",
    "online_map_paths_by_trace_index",
    "write_model_output_ass",
    "write_online_map_video",
]
