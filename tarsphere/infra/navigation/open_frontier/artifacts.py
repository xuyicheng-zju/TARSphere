"""Persist every standalone OpenFrontier intermediate for diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tarsphere.infra.navigation.open_frontier.contracts import FrontierFrame


def save_frontier_frame(frame: FrontierFrame, output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"frontier_step_{frame.observation_step:04d}"
    paths = {
        "overlay": root / f"{stem}_overlay.png",
        "processed_rgb": root / f"{stem}_rgb.png",
        "dense_arrays": root / f"{stem}_dense.npz",
        "candidates": root / f"{stem}_candidates.json",
        "graph": root / f"{stem}_graph.json",
    }
    Image.fromarray(frame.overlay_rgb, mode="RGB").save(paths["overlay"])
    Image.fromarray(frame.processed_rgb, mode="RGB").save(paths["processed_rgb"])
    np.savez_compressed(
        paths["dense_arrays"],
        processed_depth_m=frame.processed_depth_m,
        processed_camera_intrinsics=frame.processed_intrinsics.matrix(),
        normalized_distance_field=frame.prediction.normalized_distance_field,
        distance_field_px=frame.prediction.distance_field_px,
        class_indices=frame.prediction.class_indices,
        frontier_mask=frame.prediction.frontier_mask,
        information_gain=frame.prediction.information_gain,
    )
    paths["candidates"].write_text(
        json.dumps(
            {
                "observation_step": frame.observation_step,
                "processed_camera_intrinsics": (
                    frame.processed_intrinsics.matrix().tolist()
                ),
                "coordinate_conventions": {
                    "pixel_xy": "FrontierNet crop pixels: +x right, +y down",
                    "world_xyz": "Isaac world metres",
                },
                "observed_frontiers": [
                    {
                        "observation_index": item.observation_index,
                        "pixel_xy": list(item.pixel_xy),
                        "world_xyz": list(item.world_xyz),
                        "view_direction_world": list(item.view_direction_world),
                        "information_gain_m3": item.information_gain_m3,
                        "support_pixel_count": item.support_pixel_count,
                    }
                    for item in frame.observed_frontiers
                ],
                "displayed_frontiers": [
                    item.as_dict() for item in frame.displayed_frontiers
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["graph"].write_text(
        json.dumps(frame.graph_snapshot, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


__all__ = ["save_frontier_frame"]
