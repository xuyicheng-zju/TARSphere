"""Display the exact LiDAR readings retained in a model-input snapshot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


def parse_lidar_text(text):
    """Keep serialized precision and no-return semantics; never infer free sectors."""
    section = re.search(r"^###? LiDAR[^\n]*\n(.*?)(?=\n#{2,3} |\Z)", text, re.M | re.S)
    if not section:
        return None
    body = section.group(1)
    measured = re.search(r"Measured returns \(left to right\): ([^\n]+)", body)
    returns = {}
    if measured:
        for bearing, distance in re.findall(
            r"([+-]?\d+(?:\.\d+)?)°\s+(\d+(?:\.\d+)?) m", measured.group(1)
        ):
            returns[float(bearing)] = float(distance)
    clear = re.search(
        r"No return within ([\d.]+) m at: (.*?)(?:\. No sampled|\n|$)", body
    )
    limit = None
    if clear:
        limit = float(clear.group(1))
        for bearing in re.findall(r"([+-]?\d+(?:\.\d+)?)°", clear.group(2)):
            bearing = float(bearing)
            if bearing in returns:
                raise ValueError("LiDAR bearing is both a measured hit and no-return")
            returns[bearing] = None
    if not returns or 0.0 not in returns:
        raise ValueError(
            "LiDAR text lacks complete bearing evidence, including forward"
        )
    if any(
        not -180 < a <= 180 or (d is not None and (not math.isfinite(d) or d < 0))
        for a, d in returns.items()
    ):
        raise ValueError("Invalid LiDAR bearing or distance")
    return {
        "returns": [
            {"bearing_deg": a, "distance_m": returns[a]}
            for a in sorted(returns, reverse=True)
        ],
        "max_range_m": limit,
        "source": "model_input_text",
        "precision": "as serialized in the model request",
    }


def lidar_from_request(request):
    """Read actual user observation blocks, excluding system instructions/examples."""
    texts = []
    for message in request.get("request", {}).get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(x.get("text", "") for x in content if isinstance(x, dict))
    scans = [scan for text in texts if (scan := parse_lidar_text(text)) is not None]
    if len(scans) > 1:
        raise ValueError("Multiple current LiDAR sections in one model request")
    if not scans:
        return None
    return {**scans[0], "action_step": request.get("context", {}).get("action_step")}


def load_trace_lidar(trace, input_dir):
    """Resolve each inference's own archived request; never borrow another episode."""
    root = Path(input_dir)
    output = []
    for record in trace:
        scan = None
        for call in record.get("model_calls", []):
            reference = call.get("input_snapshot") or {}
            if not reference.get("path"):
                continue
            path = root / Path(reference["path"]).name
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if (
                reference.get("sha256")
                and hashlib.sha256(raw).hexdigest() != reference["sha256"]
            ):
                raise ValueError(f"Model-input hash mismatch: {path}")
            request = json.loads(raw)
            context = request.get("context", {})
            for key in ("policy_step", "action_step"):
                if key in record and key in context and record[key] != context[key]:
                    raise ValueError(f"Model-input {key} mismatch: {path}")
            found = lidar_from_request(request)
            if found is not None:
                scan = {
                    **found,
                    "request_path": str(path),
                    "request_sha256": hashlib.sha256(raw).hexdigest(),
                }
        output.append(scan)
    return output


def load_lidar_observations(*, rollout, model_input_dir, result_dir):
    """Bind every recorded inference scan to its measured pre-agent clock.

    The caller supplies the already hash-verified immutable rollout. JSON bytes
    alone suffice for LiDAR text; archived image files are never reconstructed or
    substituted. Missing scans fail this publication path instead of silently
    carrying a first-frame scan through the entire episode.
    """
    root = Path(model_input_dir).resolve()
    if not root.is_dir() or not root.is_relative_to(Path(result_dir).resolve()):
        raise ValueError("LiDAR inputs must belong to the recorded take")
    steps = {step["decision_step"]: step for step in rollout["evaluation_steps"]}
    if len(steps) != len(rollout["evaluation_steps"]):
        raise ValueError("Duplicate action indices in LiDAR observation clock")
    observations = []
    for record in rollout["trace"]:
        if not record["policy_inference"]:
            continue
        calls = record.get("model_calls") or []
        if len(calls) != 1:
            raise ValueError("LiDAR requires one recorded request per policy inference")
        reference = calls[0].get("input_snapshot") or {}
        original = Path(reference.get("path", ""))
        if original.parent.name != root.name or not re.fullmatch(
            r"\d{4,}\.request\.json", original.name
        ):
            raise ValueError("LiDAR request does not belong to this episode/attempt")
        path = (root / original.name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("LiDAR request escapes its recorded input directory")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != reference.get("sha256"):
            raise ValueError(f"LiDAR model-input hash mismatch: {path}")
        request = json.loads(raw)
        expected = {key: record[key] for key in ("policy_step", "action_step")}
        if (
            request.get("context") != expected
            or request.get("request_sha256") != reference.get("request_sha256")
            or request.get("request", {}).get("model") != calls[0].get("model")
        ):
            raise ValueError(
                "LiDAR request identity differs from its recorded inference"
            )
        scan = lidar_from_request(request)
        if scan is None:
            raise ValueError("Recorded policy request has no measured LiDAR section")
        step = steps[record["action_step"]]
        observation = step["observation"]
        clock = observation.get("camera_capture") or {}
        if (
            not step["policy_inference"]
            or observation["timing"] != "pre_agent"
            or observation["step"] != record["action_step"]
            or observation["id"] != record["images"]["pre"]
            or any(
                type(clock.get(key)) is not int or clock[key] < 0
                for key in ("physics_step", "capture_id")
            )
        ):
            raise ValueError("LiDAR requires its measured pre-agent observation clock")
        if observations and clock["physics_step"] < observations[-1]["physics_step"]:
            raise ValueError("LiDAR capture clocks are not chronological")
        observations.append(
            {
                **expected,
                **clock,
                "scan": scan,
                "request_path": str(path),
                "request_sha256": digest,
                "provider_request_sha256": reference["request_sha256"],
            }
        )
    if not observations:
        raise ValueError("No recorded LiDAR observations in this take")
    return observations


def radar_point(bearing_deg, distance_m, *, radius=74, view_range_m=2):
    """Robot forward is up; positive/left bearings go left. Radius is clearance."""
    angle = math.radians(bearing_deg)
    r = radius * min(distance_m, view_range_m) / view_range_m
    return -math.sin(angle) * r, -math.cos(angle) * r


def draw_radar(
    ass,
    start,
    end,
    x,
    y,
    scan,
    *,
    footer_text=None,
    width=228,
    paper=False,
    height=None,
    header_height=None,
):
    """Sparse polar returns, not an interpolated free-space map or fake sweep."""
    w, h = width, height if height is not None else 248
    compact = paper and h == 132
    if compact and hasattr(ass, "top_body"):
        ass.top_body(start, end, x, y, w, h)
    else:
        ass.box(start, end, x, y, w, h, alpha="35", radius=12)
    if paper:
        ass.box(
            start,
            end,
            x,
            y,
            w,
            header_height if header_height is not None else 44,
            color="purple" if compact else "teal",
            radius=0 if compact else 12,
        )
        if compact:
            for edge in (y, y + 36, y + h):
                ass.rule(start, end, x, edge, w, color="line")
    ass.text(
        start,
        end,
        x + 12 if compact else x + 14,
        y + 11 if compact else y + 12,
        "LiDAR",
        size=12 if compact else 11,
        color="purple" if compact else "teal",
        bold=True,
    )
    step = scan.get("action_step")
    ass.text(
        start,
        end,
        x + w - 64 if compact else x + w - 74,
        y + 12 if compact else y + 13,
        f"SCAN {int(step):03d}" if step is not None else "SCAN",
        size=9,
        color="muted",
    )
    cx, cy, radius = (
        x + w / 2,
        y + (76 if compact else 122),
        min(26 if compact else 80, (w - 40) / 2),
    )
    view_range = min(2.0, scan.get("max_range_m") or 2.0)

    def path(points, *, color, alpha="00", border=0, fill=True, layer=3):
        q = ass.q
        drawing = "m " + " l ".join(f"{round(a * q)} {round(b * q)}" for a, b in points)
        if fill:
            drawing += " c"
        ass.emit(
            start,
            end,
            r"{\an7\pos(0,0)\p1\shad0"
            + rf"\bord{border * q:.2f}\1c&H{color}&\3c&H{color}&"
            + rf"\1a&H{alpha if fill else 'FF'}&\3a&H{alpha}&}}"
            + drawing,
            layer,
        )

    def circle(px, py, r, color, alpha="00", filled=False, layer=3):
        points = [
            (px + r * math.cos(i * math.tau / 64), py + r * math.sin(i * math.tau / 64))
            for i in range(65)
        ]
        path(
            points,
            color=color,
            alpha=alpha,
            border=0 if filled else 0.6,
            fill=filled,
            layer=layer,
        )

    for metres in (0.5, 1.0, view_range):
        circle(
            cx,
            cy,
            radius * metres / view_range,
            "9A818E" if paper else "938271",
            "20" if paper else "80",
            layer=2,
        )
    for bearing in (0, 90, 180, -90):
        dx, dy = radar_point(
            bearing, view_range, radius=radius, view_range_m=view_range
        )
        path(
            [(cx, cy), (cx + dx, cy + dy)],
            color="9A818E" if paper else "938271",
            alpha="30" if paper else "90",
            border=0.5,
            fill=False,
            layer=2,
        )
    ass.text(
        start,
        end,
        cx - 26 if compact else cx - 16,
        cy - radius - 13,
        f"FRONT · {view_range:g} m" if compact else "FRONT",
        size=8,
        color="muted",
    )
    ass.text(start, end, cx - radius - 11, cy - 5, "L", size=9, color="muted")
    ass.text(start, end, cx + radius + 4, cy - 5, "R", size=9, color="muted")
    ass.text(start, end, cx + 3, cy - radius / 2 - 10, "1 m", size=8, color="muted")
    for beam in scan["returns"]:
        distance = beam["distance_m"]
        no_return = distance is None
        reach = view_range if no_return else distance
        dx, dy = radar_point(
            beam["bearing_deg"], reach, radius=radius, view_range_m=view_range
        )
        # BGR colors: coral close hits, warm intermediate hits, teal farther hits.
        color = (
            "889CE8"
            if distance is not None and distance < 0.4
            else ("9AC3E6" if distance is not None and distance < 1 else "BCD69C")
        )
        if paper:
            color = (
                "4D62AC"
                if distance is not None and distance < 0.4
                else ("3C77B7" if distance is not None and distance < 1 else "676F23")
            )
        path(
            [(cx, cy), (cx + dx, cy + dy)],
            color=color,
            alpha=("20" if not no_return else "70")
            if paper
            else ("65" if not no_return else "A5"),
            border=0.65,
            fill=False,
        )
        if distance is not None and distance <= view_range:
            circle(cx + dx, cy + dy, 2.1, color, filled=True, layer=4)
        else:
            circle(cx + dx, cy + dy, 1.6, color, alpha="50", filled=False, layer=4)
    # Direction icon denotes the robot; it does not mask a near-range ring.
    path(
        [(cx, cy - 5), (cx - 3.5, cy + 3), (cx, cy + 1), (cx + 3.5, cy + 3)],
        color="7A516B" if paper else "F6F3F0",
        layer=5,
    )
    detected = [b["distance_m"] for b in scan["returns"] if b["distance_m"] is not None]
    closest = f"NEAREST  {min(detected):.2f} m" if detected else "NO MEASURED HIT"
    ass.text(
        start,
        end,
        x + 12 if compact else x + 14,
        y + (106 if compact else 202 if paper else 216),
        closest,
        size=10,
        color="white",
        bold=True,
    )
    if compact:
        parts = (footer_text or "").splitlines()
        ass.text(
            start,
            end,
            x + 12,
            y + 119,
            parts[-1] if parts else "RECORDED MEASUREMENTS",
            size=8,
            color="muted",
        )
    elif paper:
        parts = (footer_text or "").splitlines()
        ass.text(
            start,
            end,
            x + 14,
            y + 218,
            f"LAST SCAN · {view_range:g} m view",
            size=8,
            color="muted",
        )
        ass.text(
            start,
            end,
            x + 14,
            y + 232,
            parts[-1] if parts else "RECORDED MEASUREMENTS",
            size=8,
            color="muted",
        )
    else:
        ass.text(
            start,
            end,
            x + 14,
            y + 232,
            footer_text or f"{view_range:g} m view · body-surface clearance",
            size=8,
            color="muted",
        )


def execution_feedback_caption(record):
    """Summarize public, measured outcomes without claiming task completion."""
    interaction = str(record.get("interaction_feedback") or "").strip()
    if interaction:
        return ("INTERACTION FEEDBACK", interaction, "teal")
    replies = [
        h
        for h in record.get("human_events", [])
        if h.get("timing") == "after_agent" and h.get("message")
    ]
    if replies:
        name = re.sub(
            r"(?<=[a-z])(?=[A-Z])", " ", str(replies[-1].get("human_id", "Human"))
        )
        return ("REPLY RECEIVED", f"{name} replied to the robot.", "teal")
    outcome = record.get("motion_outcome") or {}
    actual = outcome.get("achieved_body_delta") or {}
    requested = outcome.get("requested_body_delta") or {}
    parts = []
    if actual and requested and outcome.get("available"):
        forward = float(requested.get("forward_m", 0))
        if abs(forward) > 0.001:
            parts.append(
                f"Moved {float(actual.get('forward_m', 0)):.2f} m / requested {forward:g} m"
            )
        lateral = float(requested.get("left_m", 0))
        if abs(lateral) > 0.001:
            parts.append(
                f"Lateral {float(actual.get('left_m', 0)):.2f} m / requested {lateral:g} m"
            )
        turn = float(requested.get("turn_deg", 0))
        if abs(turn) > 0.001:
            measured = float(actual.get("turn_deg", 0))
            parts.append(
                f"Turned {abs(measured):.1f}° {'left' if measured >= 0 else 'right'} / requested {abs(turn):g}°"
            )
    contact = bool(record.get("physical_contact"))
    if contact:
        details = record.get("physical_contact_details") or []
        bearings = [
            float(d["relative_bearing_deg"])
            for d in details
            if d.get("relative_bearing_deg") is not None
        ]
        direction = ""
        if bearings:
            a = bearings[0]
            direction = (
                " ahead"
                if abs(a) < 25
                else (
                    " behind"
                    if abs(a) > 150
                    else (" on left" if a > 0 else " on right")
                )
            )
        parts.insert(0, "Contact" + direction)
    if parts:
        return ("EXECUTION FEEDBACK", " · ".join(parts), "coral" if contact else "teal")
    text = str(record.get("execution_feedback") or "").strip()
    if text:
        return ("EXECUTION FEEDBACK", text, "teal")
    return None
