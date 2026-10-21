"""Original floating-card appearance on the validated native ego timeline."""

from dataclasses import dataclass
from pathlib import Path
import re
from bisect import bisect_right
import hashlib
from .model_output_overlay import (
    _FocusAss,
    _action_caption,
    _current_task_text,
    _HUD_COLORS,
    _escape_ass,
)


def _frame_boundary_seconds(frame_index):
    """Place a 30 Hz ASS boundary strictly before its source frame PTS.

    libass consumes truncated milliseconds. An exact centisecond boundary can
    therefore miss the first frame of a one-frame decision after FFmpeg's PTS
    rescaling. The preceding centisecond is still after the preceding frame.
    """
    return max(0, (int(frame_index) * 100 - 1) // 30) / 100


class _PaperHudAss(_FocusAss):
    """Local paper palette; other presentation modes keep their original colours."""

    def __init__(self, width, height):
        super().__init__(width, height, font_family="Times New Roman")

    palette = dict(
        white="302A32",
        muted="5D4B58",
        purple="7A516B",
        teal="505B24",
        ink="FBF9FC",
        line="BEA2CC",
    )

    def text(self, *args, **kwargs):
        color = kwargs.get("color", "white")
        first = len(self.events)
        super().text(*args, **kwargs)
        x = args[2] if len(args) > 2 else kwargs.get("x", 0)
        y = args[3] if len(args) > 3 else kwargs.get("y", 0)
        body = x < 960 and 52 <= y < 148
        fill = (
            self.palette["white"]
            if body
            else self.palette.get(color, self.palette["white"])
        )
        for i in range(first, len(self.events)):
            self.events[i] = self.events[i].replace(_HUD_COLORS[color], fill)

    def box(self, *args, **kwargs):
        color = kwargs.get("color", "ink")
        # Large ego cards transmit the real scene; rules/icons and sidebar remain solid.
        x = args[2] if len(args) > 2 else kwargs.get("x", 0)
        w = args[4] if len(args) > 4 else kwargs.get("w", 0)
        h = args[5] if len(args) > 5 else kwargs.get("h", 0)
        if x < 960:
            kwargs["radius"] = 0
        kwargs["alpha"] = "70" if x < 960 and w > 30 and h > 4 else "00"
        first = len(self.events)
        super().box(*args, **kwargs)
        fills = {
            "ink": "FBF9FC",
            "purple": "F1DCEB",
            "teal": "E7EED5",
            "line": "BEA2CC",
        }
        for i in range(first, len(self.events)):
            self.events[i] = self.events[i].replace(
                _HUD_COLORS[color], fills.get(color, "FBF9FC")
            )

    def top_body(self, start, end, x, y, w, h):
        # Keep the original underlay below the unchanged purple header.
        self.box(start, end, x, y, w, 36, radius=0)
        first = len(self.events)
        super().box(start, end, x, y + 36, w, h - 36, color="ink", alpha="70", radius=0)
        # One light paper body, matching opacity in all three columns.
        for i in range(first, len(self.events)):
            self.events[i] = self.events[i].replace(_HUD_COLORS["ink"], "F7EEF3")

    def title(self, *args, **kwargs):
        first = len(self.events)
        self.text(*args, **kwargs)
        for i in range(first, len(self.events)):
            self.events[i] = self.events[i].replace(
                "{\\an7", "{\\fnTimes New Roman\\an7"
            )


@dataclass(frozen=True)
class FullHudLayout:
    width: int
    ego_height: int

    @property
    def sidebar(self):
        return round(self.ego_height * 4 / 9 / 2) * 2

    @property
    def resolution(self):
        return self.width + self.sidebar, self.ego_height

    def rect(self, name):
        if name == "ego":
            return 0, 0, self.width, self.ego_height
        if name == "map":
            q = self.ego_height / 720
            return tuple(
                round(x / 2) * 2
                for x in (self.width + 16 * q, 16 * q, self.sidebar - 32 * q, 248 * q)
            )
        if name == "sensor_image":
            x, y, width, height = self.rect("map")
            image_height = round(width * 3 / 4)
            return x, y + (height - image_height) // 2, width, image_height
        raise KeyError(name)


def _terminal_feedback(recorded_result):
    if recorded_result.get("success") is True:
        return "Task completed", "teal"
    if recorded_result.get("termination_reason") == "safety_violation" and any(
        v.get("type") == "robot_fall"
        for v in (recorded_result.get("safety_violations") or [])
    ):
        return "Safety stop · Robot fell", "purple"
    if recorded_result.get("outcome_status") == "infrastructure_invalid":
        return "Recording interrupted", "muted"
    if recorded_result.get("outcome_status") == "pending_evaluation":
        if recorded_result.get("termination_reason") == "object_dropped":
            return "Object dropped · Evaluation pending", "purple"
        return "Evaluation pending", "purple"
    if recorded_result.get("outcome_status") == "scored_pending_attribution":
        return "Task not completed · Review pending", "purple"
    if recorded_result.get("success") is False:
        return "Task not completed", "purple"
    return "Outcome unavailable", "muted"


def _reasoning_window(ass, thinking, quote, width, max_lines):
    """Select continuous original characters, retaining the preceding sentence."""

    def display(begin, end):
        return (
            ("… " if begin else "")
            + thinking[begin:end]
            + (" …" if end < len(thinking) else "")
        )

    def fits(begin, end):
        return len(ass.wrap(display(begin, end), width, 16, 10000)) <= max_lines

    full = " ".join(ass.wrap(thinking, width, 16, max_lines))
    clean_quote = " ".join(_escape_ass(quote).split())
    if fits(0, len(thinking)):
        return thinking, 0, len(thinking), False
    begin = thinking.index(quote)
    end = begin + len(quote)
    if not fits(begin, end):
        raise ValueError(
            "Editorial thinking quote cannot fit at the original Reasoning font size"
        )
    sentence_starts = [0] + [
        m.end() for m in re.finditer(r"(?<=[.!?])\s+", thinking[:begin])
    ]
    current_start = sentence_starts[-1]
    previous_start = sentence_starts[-2] if len(sentence_starts) > 1 else current_start
    if clean_quote in full and fits(0, end):
        begin = 0
    elif fits(previous_start, end):
        begin = previous_start
    elif fits(current_start, end):
        begin = current_start
    # Extend the continuous window by whole original words, never summarise.
    boundaries = [(m.start(), m.end()) for m in re.finditer(r"\S+", thinking)]
    for _, candidate in boundaries:
        if candidate > end and fits(begin, candidate):
            end = candidate
        elif candidate > end:
            break
    for candidate, _ in reversed(boundaries):
        if candidate < begin and fits(candidate, end):
            begin = candidate
        elif candidate < begin:
            break
    return display(begin, end), begin, end, True


def _quoted_reasoning(ass, a, b, x, y, thinking, width, *, max_lines, annotation):
    """Underline an exact quote; excerpts are continuous recorded model text."""
    displayed, source_begin, source_end, excerpt = _reasoning_window(
        ass, thinking, annotation["thinking_quote"], width, max_lines
    )
    lines = ass.wrap(displayed, width, 16, max_lines)
    visible = " ".join(lines)
    quote = " ".join(_escape_ass(annotation["thinking_quote"]).split())
    offset = visible.find(quote)
    if offset < 0:
        raise ValueError(
            "Editorial quote is not completely visible in the original-text window"
        )
    if excerpt:
        ass.title(
            a, b, x, y - 21, "Reasoning · excerpt", size=16, color="purple", bold=True
        )
    else:
        ass.title(a, b, x, y - 21, "Reasoning", size=16, color="purple", bold=True)
    cursor = 0
    for index, line in enumerate(lines):
        begin, finish = (
            max(0, offset - cursor),
            min(len(line), offset + len(quote) - cursor),
        )
        text = line
        if finish > begin:
            text = (
                line[:begin]
                + r"{\u1\c&H"
                + ass.palette["purple"]
                + "&}"
                + line[begin:finish]
                + r"{\u0\c&H"
                + ass.palette["white"]
                + "&}"
                + line[finish:]
            )
        ass.emit(
            a,
            b,
            r"{\an7"
            + ass.position(x, y + index * 20)
            + rf"\fs{16 * ass.q:.2f}\c&H{ass.palette['white']}&\b0\q2}}"
            + text,
        )
        cursor += len(line) + 1
    note_lines = ass.wrap(annotation["note"], width, 12, 100)
    if len(note_lines) > 3:
        raise ValueError(
            "Editorial review note must fit the reserved three-line footer"
        )
    ass.rule(a, b, x, 540, width, color="line")
    ass.text(a, b, x, 548, "REVIEW NOTE", size=10.5, color="purple", bold=True)
    ass.paragraph(
        a, b, x, 564, annotation["note"], width, size=12, max_lines=3, color="muted"
    )
    return {
        "status": "shown",
        "thinking_sha256": hashlib.sha256(thinking.encode("utf-8")).hexdigest(),
        "excerpt": excerpt,
        "source_text_start": source_begin,
        "source_text_end_exclusive": source_end,
        "display_text": displayed,
    }


def _empty_radar(ass, a, b, ego_width):
    x = ego_width - 160
    ass.top_body(a, b, x, 16, 144, 132)
    ass.box(a, b, x, 16, 144, 36, color="purple", radius=0)
    for edge in (16, 52, 148):
        ass.rule(a, b, x, edge, 144, color="line")
    ass.text(a, b, x + 12, 27, "LiDAR", size=12, color="purple", bold=True)
    ass.paragraph(
        a,
        b,
        x + 12,
        60,
        "Awaiting recorded LiDAR",
        120,
        size=16,
        max_lines=4,
        color="muted",
    )


def write_full_hud_ass(
    *,
    path,
    layout,
    states,
    task_title,
    source_label,
    recorded_result,
    terminal_frame_index,
    lidar_observations=None,
    physics_steps=None,
    model_observations=None,
    grounding_depth_observations=None,
    annotation_report=None,
):
    """Style existing causal states; never derive an earlier terminal outcome."""
    ass = _PaperHudAss(*layout.resolution)
    q = ass.q
    ew = layout.width / q
    sx, sw = ew + 16, layout.sidebar / q - 32
    end = (len(states) * 100 // 30) / 100
    ass.rule(0, end, sx, 282, sw, color="purple", thickness=2)
    ass.box(0, end, sx, 620, sw, 84, alpha="00", radius=0)
    ass.paragraph(
        0, end, sx + 12, 684, source_label, 140, size=10, max_lines=1, color="muted"
    )
    if physics_steps is not None:
        if len(physics_steps) != len(states) or any(
            b < a for a, b in zip(physics_steps, physics_steps[1:])
        ):
            raise ValueError("HUD physics_steps must match states and be chronological")
        for i in range(0, len(states), 3):
            ass.text(
                _frame_boundary_seconds(i),
                end if i + 3 >= len(states) else _frame_boundary_seconds(i + 3),
                sx + sw - 112,
                684,
                f"SIM {physics_steps[i] / 240:07.2f} s",
                size=10,
                color="muted",
            )
    if model_observations and grounding_depth_observations:
        raise ValueError("The full HUD sensor card has exactly one evidence source")
    depth_panel = bool(grounding_depth_observations)
    sensor_observations = (
        grounding_depth_observations if depth_panel else model_observations
    )
    if sensor_observations:
        if physics_steps is None:
            raise ValueError("Recorded sensor card requires source physics_steps")
        ticks = [item.physics_step for item in sensor_observations]
        if any(type(tick) is not int or tick < 0 for tick in ticks) or any(
            b < a for a, b in zip(ticks, ticks[1:])
        ):
            raise ValueError(
                "Recorded sensor capture ticks must be chronological integers"
            )
        title = "CONDITIONED DEPTH" if depth_panel else "MODEL OBSERVATION"
        ass.text(
            0,
            end,
            sx + 6,
            18,
            title + " · 640 × 480",
            size=10.5,
            color="purple",
            bold=True,
        )
        description = (
            "Conditioned depth for grounding tools"
            if depth_panel
            else "Actual image sent to the model"
        )
        ass.text(0, end, sx + 6, 268, description, size=9.5, color="muted")
        _, image_y, _, image_height = layout.rect("sensor_image")
        footer_y = (image_y + image_height) / q + 2
        indices = [bisect_right(ticks, tick) - 1 for tick in physics_steps]
        first = 0
        while first < len(indices):
            last = first + 1
            while last < len(indices) and indices[last] == indices[first]:
                last += 1
            a, b = (
                _frame_boundary_seconds(first),
                end if last == len(states) else _frame_boundary_seconds(last),
            )
            index = indices[first]
            if index < 0:
                ass.paragraph(
                    a,
                    b,
                    sx + 12,
                    120,
                    "Awaiting recorded depth"
                    if depth_panel
                    else "Awaiting recorded model input",
                    sw - 24,
                    size=15,
                    max_lines=2,
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
                    a,
                    b,
                    sx + 6,
                    footer_y,
                    f"{tag} · CAPTURE {capture.physics_step / 240:.2f} s",
                    size=10,
                    color="teal",
                    bold=True,
                )
            first = last
    if lidar_observations is None:
        _empty_radar(ass, 0, end, ew)
    else:
        if physics_steps is None:
            raise ValueError("Recorded LiDAR requires source physics_steps")
        ticks = [item["physics_step"] for item in lidar_observations]
        if any(not isinstance(t, int) for t in ticks) or any(
            b < a for a, b in zip(ticks, ticks[1:])
        ):
            raise ValueError(
                "Recorded LiDAR capture ticks must be chronological integers"
            )
        indices = [bisect_right(ticks, tick) - 1 for tick in physics_steps]
        first = 0
        from .presentation_radar import draw_radar

        while first < len(indices):
            last = first + 1
            while last < len(indices) and indices[last] == indices[first]:
                last += 1
            a, b = (
                _frame_boundary_seconds(first),
                end if last == len(states) else _frame_boundary_seconds(last),
            )
            index = indices[first]
            if index < 0:
                _empty_radar(ass, a, b, ew)
            else:
                item = lidar_observations[index]
                scan = dict(item.get("scan") or item)
                scan["action_step"] = item.get("action_step", scan.get("action_step"))
                draw_radar(
                    ass,
                    a,
                    b,
                    ew - 160,
                    16,
                    scan,
                    width=144,
                    paper=True,
                    height=132,
                    header_height=36,
                    footer_text=f"LAST SCAN\nCAPTURE {ticks[index] / 240:.2f} s",
                )
            first = last
    initial = next(
        (s["plan"].goal for s in states if s["plan"] and s["plan"].goal), None
    )
    start = 0
    while start < len(states):
        stop = start + 1
        while stop < len(states) and states[stop] == states[start]:
            if recorded_result and stop == terminal_frame_index:
                break
            stop += 1
        a, b = (
            _frame_boundary_seconds(start),
            end if stop == len(states) else _frame_boundary_seconds(stop),
        )
        s = states[start]
        plan, decision = s["plan"], s["decision"]
        final = (
            recorded_result
            and terminal_frame_index is not None
            and start >= terminal_frame_index
        )
        goal = plan.goal if plan and plan.goal else task_title
        if task_title and goal == initial:
            goal = task_title
        gw = 324
        # One continuous top band: all three columns share top/header/footer.
        ass.top_body(a, b, 16, 16, gw, 132)
        ass.box(a, b, 16, 16, gw, 36, color="purple", radius=0)
        ass.rule(a, b, 16, 16, gw, color="line")
        ass.rule(a, b, 16, 52, gw, color="line")
        ass.rule(a, b, 16, 148, gw, color="line")
        ass.text(a, b, 28, 27, "Goal", size=12, color="purple", bold=True)
        ass.paragraph(a, b, 80, 22, goal, gw - 76, size=11.5, max_lines=2)
        if decision:
            current = _current_task_text(decision, plan, s["archive_schema"])
            ass.paragraph(
                a, b, 28, 60, current, gw - 24, size=16, max_lines=2, bold=True
            )
            active = min(max(decision.action_index, 0), len(decision.actions) - 1)
            tool, args = decision.actions[active]
            label, detail = _action_caption(tool, args)
            argument_label = re.search(r'(?<!\w)label\s*=\s*"([^"\n]+)"', args)
            frontier = (
                argument_label
                if tool in ("navigate_to_frontier", "explore_frontier")
                else None
            )
            if tool == "remember_entity" and argument_label:
                detail = "Anchor: " + argument_label.group(1)
            description = re.search(
                r'(?<!\w)target_description\s*=\s*"([^"\n]+)"', args
            )
            if frontier or description:
                detail = " · ".join(
                    x
                    for x in [
                        ("Frontier " + frontier.group(1)) if frontier else "",
                        description.group(1) if description else "",
                    ]
                    if x
                )
            ass.text(
                a,
                b,
                sx,
                302,
                "LAST ACTION" if final else "NOW",
                size=10.5,
                color="purple",
                bold=True,
            )
            ass.text(
                a,
                b,
                sx + sw - 90,
                302,
                f"ACTION {active + 1:02d}/{max(decision.sequence_length, len(decision.actions)):02d}",
                size=10,
                color="muted",
            )
            used = ass.paragraph(
                a, b, sx, 323, label, sw, size=25, max_lines=2, bold=True
            )
            ass.paragraph(
                a, b, sx, 331 + used, detail, sw, size=14, max_lines=2, color="muted"
            )
            reason_y = max(399, 331 + used + 56)
            ass.rule(a, b, sx, reason_y - 15, sw)
            max_lines = max(3, min(7, int((610 - reason_y - 21) / 20)))
            annotation = s.get("editorial_annotation")
            if annotation:
                max_lines = max(1, min(max_lines, int((538 - reason_y - 21) / 20)))
                if (
                    decision.policy_step != annotation["policy_step"]
                    or annotation["thinking_quote"] not in decision.thinking
                ):
                    raise ValueError(
                        "Editorial annotation does not match the displayed recorded thinking"
                    )
                report = _quoted_reasoning(
                    ass,
                    a,
                    b,
                    sx,
                    reason_y + 21,
                    decision.thinking,
                    sw,
                    max_lines=max_lines,
                    annotation=annotation,
                )
                if "thinking_sha256" in annotation:
                    report["thinking_sha256"] = annotation["thinking_sha256"]
                    offset = annotation["thinking_display_offset"]
                    report["source_text_start"] += offset
                    report["source_text_end_exclusive"] += offset
                if annotation_report is not None:
                    annotation_report.append(
                        {
                            "action_step": annotation["action_step"],
                            "policy_step": annotation["policy_step"],
                            "frame_start": start,
                            "frame_end_exclusive": stop,
                            "displayed_action_step": s.get(
                                "editorial_display_action_step",
                                annotation["action_step"],
                            ),
                            **(
                                {
                                    "source_physics_start": physics_steps[start],
                                    "source_physics_end": physics_steps[stop - 1],
                                }
                                if physics_steps is not None
                                else {}
                            ),
                            **report,
                        }
                    )
            else:
                ass.title(
                    a, b, sx, reason_y, "Reasoning", size=16, color="purple", bold=True
                )
                ass.paragraph(
                    a,
                    b,
                    sx,
                    reason_y + 21,
                    decision.thinking,
                    sw,
                    size=16,
                    max_lines=max_lines,
                )
        feedback = s["feedback"]
        label, detail, color = (
            ("Result", *_terminal_feedback(recorded_result))
            if final
            else (
                "Feedback",
                feedback[1] if feedback else "No recent recorded feedback",
                "teal",
            )
        )
        detail = detail.replace("_", " ")
        if detail:
            detail = detail[0].upper() + detail[1:]
        technical = re.search(r"\s*\(score\b|\s*pixel box\s*\[", detail)
        if technical:
            detail = detail[: technical.start()].rstrip() + " …"
        ass.rule(a, b, 16, 104, gw, color="line")
        ass.paragraph(
            a,
            b,
            28,
            112,
            label + " · " + detail,
            gw - 24,
            size=12,
            max_lines=2,
            color=color,
        )
        speech = s["speech"]
        cx, cw, cy, ch = 340, ew - 500, 16, 132
        ass.top_body(a, b, cx, cy, cw, ch)
        ass.box(a, b, cx, cy, cw, 36, color="purple", radius=0)
        ass.rule(a, b, cx, cy, cw, color="line")
        ass.rule(a, b, cx, cy + 36, cw, color="line")
        ass.rule(a, b, cx, cy + ch, cw, color="line")
        ass.rule(a, b, cx, 16, 1, thickness=132, color="line")
        ass.rule(a, b, ew - 160, 16, 1, thickness=132, color="line")
        ass.text(a, b, cx + 12, 27, "Dialogue", size=12, color="purple", bold=True)
        if speech:
            ass.text(
                a,
                b,
                cx + 92,
                27,
                "· " + speech[0] + " · last recorded",
                size=12,
                color="teal",
                bold=True,
            )
            ass.paragraph(a, b, cx + 12, 60, speech[1], cw - 24, size=16, max_lines=4)
        else:
            ass.text(
                a, b, cx + 12, 60, "No recent recorded exchange", size=16, color="muted"
            )
        start = stop
    Path(path).write_text(ass.header + "\n".join(ass.events) + "\n")
    return Path(path)
