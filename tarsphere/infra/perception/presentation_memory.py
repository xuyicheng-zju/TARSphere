"""Read-only, physics-clocked evidence for the public demo presentation.

These artifacts never enter model prompts. Only committed agent memory and
sensor-derived online occupancy are admitted; no scene ground truth is read.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

_last_map_capture: WeakKeyDictionary = WeakKeyDictionary()


def capture_voxel_snapshot(grid: Any, root: Path, physics_step: int) -> None:
    """Archive an observed map at most twice per simulation second (240 Hz)."""
    if grid.map_source != "online":
        return
    root = Path(root) / "presentation" / "voxels"
    stamp = (str(root), int(physics_step) // 120)
    if _last_map_capture.get(grid) == stamp and root.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / f"{int(physics_step):012d}.npz",
        physics_step=int(physics_step),
        map_source="online",
        grid=grid._grid,
        observed=grid._observed,
        voxel_occupied=grid._voxel_occupied,
        origin_xy=grid._origin,
        resolution_m=grid.resolution,
        voxel_z_resolution_m=grid.voxel_z_resolution_m,
        obstacle_z_range_m=[grid.obstacle_z_min_m, grid.obstacle_z_max_m],
        pose_history_xy=np.asarray(grid.exploration_pose_history()).reshape(-1, 2),
    )
    _last_map_capture[grid] = stamp


def capture_agent_memory(
    memory: Any, root: Path, physics_step: int, action_step: int
) -> None:
    """Save actual committed memory, with immutable ego crops for saved anchors."""
    from PIL import Image

    root = Path(root) / "presentation" / "memory"
    root.mkdir(parents=True, exist_ok=True)
    anchors = []
    for record in memory.semantic.landmarks:
        fingerprint = hashlib.sha256(
            f"{record.entity_id}:{record.capture_id}:{record.observed_action_step}".encode()
        ).hexdigest()[:20]
        image_path = root / f"anchor-{fingerprint}.png"
        if not image_path.exists():
            image = Image.fromarray(
                np.asarray(record.keyframe_rgb, dtype=np.uint8)[..., :3]
            )
            if record.bbox_2d:
                x0, y0, x1, y1 = record.bbox_2d
                box = (
                    max(0, int(x0)),
                    max(0, int(y0)),
                    min(image.width, int(x1)),
                    min(image.height, int(y1)),
                )
                if box[2] > box[0] and box[3] > box[1]:
                    image = image.crop(box)
            image.save(image_path)
        anchors.append(
            {
                "entity_id": record.entity_id,
                "label": record.label,
                "description": record.description,
                "entity_type": record.entity_type,
                "world_xyz": list(record.world_xyz),
                "capture_id": record.capture_id,
                "observed_action_step": record.observed_action_step,
                "image": image_path.name,
            }
        )
    visual_history = []
    visual = getattr(memory, "visual", None)
    for frame in visual.retrieved_frames() if visual is not None else ():
        for view_name, view in frame.views.items():
            rgb = getattr(view, "rgb", None)
            if rgb is None:
                continue
            name = hashlib.sha256(
                f"{frame.action_step}:{view_name}".encode()
            ).hexdigest()[:16]
            target = root / f"history-{name}.png"
            if not target.exists():
                Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).save(target)
            visual_history.append(
                {
                    "action_step": frame.action_step,
                    "view": str(view_name),
                    "image": target.name,
                }
            )
    plan = memory.strategy_planner.active_plan
    payload = {
        "physics_step": int(physics_step),
        "action_step": int(action_step),
        "key_memories": [asdict(x) for x in memory.key.records],
        "anchors": anchors,
        "visual_history": visual_history,
        "grounding": [
            asdict(x)
            for x in getattr(getattr(memory, "grounding", None), "entries", {}).values()
        ],
        "grasped_hands": [asdict(x) for x in getattr(memory.text, "grasped_hands", [])],
        "plan": plan.as_dict() if plan else None,
        "plan_history": [x.as_dict() for x in memory.strategy_planner.history],
        "text_history": [asdict(x) for x in memory.text.retrieved_records("all")],
        "current_text": asdict(memory.text.current_record)
        if memory.text.current_record
        else None,
        "humans": [asdict(x) for x in memory.human.retrieved_humans()],
    }
    (root / f"{int(physics_step):012d}-{int(action_step):06d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    )


def snapshot_at(paths: list[Path], steps: list[int], physics_step: int) -> Path | None:
    """Latest past snapshot; never display a future observation early."""
    index = bisect_right(steps, int(physics_step)) - 1
    return paths[index] if index >= 0 else None


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, max(10, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def render_voxel_projection(snapshot: Path, resolution: tuple[int, int], *, anchors=()):
    """Fixed-camera stacked 3-D occupied voxels and true 2-D occupancy.

    The framing is fixed to the declared map extent, never future extents.
    Colors indicate height/occupancy; they do not claim semantic labels or RGB.
    """
    from PIL import Image, ImageDraw

    with np.load(snapshot, allow_pickle=False) as data:
        if str(data["map_source"].item()) != "online":
            raise ValueError("Presentation voxels require sensor-derived online maps")
        grid = data["grid"].copy()
        bits = data["voxel_occupied"].astype(np.uint64)
        origin = data["origin_xy"]
        res = float(data["resolution_m"])
        zr = float(data["voxel_z_resolution_m"])
        zlo, zhi = data["obstacle_z_range_m"]
        poses = data["pose_history_xy"].reshape(-1, 2)
    w, h = resolution
    image = Image.new("RGB", (w, h), "#FAF8FC")
    draw = ImageDraw.Draw(image)
    f = _font(w / 31)
    draw.text((14, 9), "3D VOXELS", font=f, fill="#8557A0")
    draw.text((14, int(h * 0.56)), "2D PROJECTION", font=f, fill="#5C9E8D")
    n, m = grid.shape
    # Oblique projection with a fixed basis and a fixed world scale.
    k = min(w / (n + m + 6), h * 0.48 / (0.34 * (n + m) + 2 * (zhi - zlo) / res + 8))
    cx, cy = w * 0.5, h * 0.30

    def iso(r, c, z=0):
        return (
            cx + (c - r) * k,
            cy + (c + r - (n + m) / 2) * k * 0.34 - z / res * k * 0.8,
        )

    ground = [iso(0, 0), iso(n, 0), iso(n, m), iso(0, m)]
    draw.polygon(ground, fill="#EEEBF2", outline="#D8D0E2")
    # At presentation resolution aggregate neighboring XY cells by bitwise OR.
    stride = max(1, int(np.ceil(max(n, m) / 130)))
    for r in range(0, n, stride):
        for c in range(0, m, stride):
            if np.any(grid[r : r + stride, c : c + stride] == -1):
                draw.polygon(
                    [
                        iso(r, c),
                        iso(r + stride, c),
                        iso(r + stride, c + stride),
                        iso(r, c + stride),
                    ],
                    fill="#DCEDE6",
                )
    padded = np.pad(bits, ((0, (-n) % stride), (0, (-m) % stride)))
    coarse = np.bitwise_or.reduce(
        np.bitwise_or.reduce(
            padded.reshape(
                padded.shape[0] // stride, stride, padded.shape[1] // stride, stride
            ),
            axis=3,
        ),
        axis=1,
    )
    rr, cc = np.nonzero(coarse)
    order = np.argsort(rr + cc, kind="stable")
    zcount = min(63, int(np.ceil((zhi - zlo) / zr)))
    for index in order:
        r, c = int(rr[index]), int(cc[index])
        value = int(coarse[r, c])
        for z in range(zcount):
            if not value & (1 << z):
                continue
            a, b = r * stride, c * stride
            ztop = zlo + (z + 1) * zr
            zbottom = zlo + z * zr
            corners = [
                iso(a, b, ztop),
                iso(a + stride, b, ztop),
                iso(a + stride, b + stride, ztop),
                iso(a, b + stride, ztop),
            ]
            low = [
                iso(a + stride, b, zbottom),
                iso(a + stride, b + stride, zbottom),
                iso(a, b + stride, zbottom),
            ]
            draw.polygon([corners[1], corners[2], low[1], low[0]], fill="#745494")
            draw.polygon([corners[2], corners[3], low[2], low[1]], fill="#5C9E8D")
            draw.polygon(corners, fill="#BBA3CC")
    # Same map origin, +X upward and +Y left, matching the agent map convention.
    size = int(min(w * 0.8, h * 0.35))
    x0 = (w - size) // 2
    y0 = int(h * 0.63)
    rgb = np.full((*grid.shape, 3), [239, 236, 243], dtype=np.uint8)
    rgb[grid == -1] = [220, 237, 230]
    rgb[grid == 1] = [116, 84, 148]
    bev = Image.fromarray(rgb[::-1, ::-1]).resize(
        (size, size), Image.Resampling.NEAREST
    )
    image.paste(bev, (x0, y0))
    draw = ImageDraw.Draw(image)

    def project(xy):
        rc = (np.asarray(xy) - origin) / res
        return (x0 + (m - 1 - rc[1]) * size / m, y0 + (n - 1 - rc[0]) * size / n)

    if len(poses) > 1:
        draw.line([project(xy) for xy in poses], fill="#5C9E8D", width=max(1, w // 180))
    if len(poses):
        x, y = project(poses[-1])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#8557A0", outline="white")
    for number, anchor in enumerate(anchors, 1):
        xy = anchor["world_xyz"][:2]
        x, y = project(xy)
        if x0 <= x < x0 + size and y0 <= y < y0 + size:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#CC9277", outline="white")
            draw.text((x + 6, y - 6), str(number), font=_font(w / 40), fill="#8557A0")
            rc = (np.asarray(xy) - origin) / res
            vx, vy = iso(rc[0], rc[1], float(anchor["world_xyz"][2]))
            draw.ellipse(
                (vx - 4, vy - 4, vx + 4, vy + 4), fill="#CC9277", outline="white"
            )
            draw.text((vx + 5, vy - 8), str(number), font=_font(w / 36), fill="#8557A0")
    return np.asarray(image)


def memory_changes(previous: dict, current: dict) -> list[dict]:
    """Diff committed ledger versions, never proposed/rejected model updates."""
    old = {str(x["memory_id"]): x for x in previous.get("key_memories", [])}
    new = {str(x["memory_id"]): x for x in current.get("key_memories", [])}
    changes = []
    for key, value in new.items():
        if key not in old:
            changes.append(
                {"op": "REMEMBER", "id": key, "before": "", "after": value["content"]}
            )
        elif value != old[key]:
            changes.append(
                {
                    "op": "REVISE",
                    "id": key,
                    "before": old[key]["content"],
                    "after": value["content"],
                }
            )
    for key in old.keys() - new.keys():
        changes.append(
            {"op": "FORGET", "id": key, "before": old[key]["content"], "after": ""}
        )
    return changes
