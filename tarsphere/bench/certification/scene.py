"""Load certified scene navigation evidence for a compiled episode."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from functools import lru_cache
from typing import Any

from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.assets import tarsphere_assets_root


@lru_cache(maxsize=16)
def _official_runtime_navigation_mask(path, source_sha256, radius, bodies):
    """One runtime raster/inflation policy, with official-axis padding restored."""
    import numpy as np
    from tarsphere.infra.perception.online_mapping import OnlineOccupancyGrid

    grid = OnlineOccupancyGrid.from_official_occupancy(
        path, robot_radius_m=radius, static_obstacles=bodies
    )
    native = grid.planning_navigable_mask()
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != source_sha256:
        raise ValueError(
            "Occupancy changed while computing the runtime navigation mask"
        )
    padded = np.zeros((native.shape[0] + 1, native.shape[1] + 1), dtype=bool)
    padded[1:, 1:] = native[:, ::-1].T
    return padded


def official_runtime_navigation_mask(
    occupancy_path, *, clearance_radius_m, static_obstacles=()
):
    """Runtime body raster + G1 erosion, in OfficialOccupancyBevMap coordinates.

    Callers pass this as a prepared Planner mask. Do not add the same Humans
    again as world exclusion zones or inflate this already-clear mask again.
    Source freshness remains the caller's publication/provenance obligation.
    """
    path = Path(occupancy_path).resolve()
    # Equal-size rapid rewrites can retain the same stat timestamp, including
    # on the simulation host. Metadata-only cache identities can reopen walls.
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    bodies = tuple(tuple(float(value) for value in body) for body in static_obstacles)
    return _official_runtime_navigation_mask(
        str(path), source_sha256, float(clearance_radius_m), bodies
    ).copy()


def _file_digest(path: Path) -> str:
    """Hash current bytes; size and timestamp do not prove source freshness."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_scene_occupancy_provenance(
    path: Path,
    *,
    root: Path,
    scene_usd: Path,
    verify_scene_files: bool = True,
) -> dict:
    """Reject shipped/stale maps and stale composed geometry before use."""
    provenance = json.loads(path.with_suffix(".json").read_text())
    if provenance.get("schema_version") != "tarsphere.navigation_occupancy.v2":
        raise ValueError(f"{path}: rebake from the current evaluation USD is required")
    if (
        provenance.get("uncovered_collision_roots") != []
        or provenance.get("composition_error_count") != 0
    ):
        raise ValueError(f"{path}: incomplete collision composition/coverage")
    if (root / provenance["source"]).resolve() != scene_usd.resolve():
        raise ValueError(f"{path}: occupancy belongs to a different evaluation USD")
    evidence = [(path, provenance["occupancy_sha256"])]
    layers = provenance.get("composed_layers")
    if not layers:
        raise ValueError(f"{path}: missing composed geometry hashes")
    # Portable authoring workspaces carry baked evidence, not the complete USD
    # asset library. Runtime must additionally verify every composed layer.
    if verify_scene_files:
        evidence.append((scene_usd, provenance["source_sha256"]))
        evidence.extend((root / row["path"], row["sha256"]) for row in layers)
    for file, expected in evidence:
        if _file_digest(file) != expected:
            raise ValueError(f"{path}: stale occupancy evidence for {file}")
    return provenance


def load_episode_occupancy(
    episode: Any,
    *,
    assets_root: Path | None = None,
) -> OfficialOccupancyBevMap:
    """Load the exact scene occupancy declared by compiled scene facts."""
    world = getattr(episode, "world_knowledge", None)
    if world is None and isinstance(episode, dict):
        world = episode.get("world_knowledge")
    world = world or {}
    scene_map = world.get("scene_map")
    if not isinstance(scene_map, dict):
        raise ValueError("Compiled episode requires world_knowledge.scene_map")
    relative_path = str(scene_map.get("occupancy") or "").strip()
    if not relative_path:
        raise ValueError(
            "Compiled episode requires world_knowledge.scene_map.occupancy"
        )
    path = Path(relative_path).expanduser()
    if not path.is_absolute():
        path = (assets_root or tarsphere_assets_root()) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Compiled episode occupancy is missing: {path}")
    scene = getattr(episode, "scene", None)
    scene_asset = str(getattr(scene, "asset_path", "") or "")
    if scene_asset.endswith("/start_result_navigation.usd"):
        assets = assets_root or tarsphere_assets_root()
        validate_scene_occupancy_provenance(
            path, root=assets.parent, scene_usd=assets / scene_asset
        )
    return OfficialOccupancyBevMap(occupancy_path=path)


def load_episode_navigation_grid(episode: Any):
    """Shared GT navigation grid, including authored static Human bodies.

    The scene bake predates episode NPC spawning. Their actual body radius is
    occupied geometry; the runtime planner adds its usual 0.25 m clearance.
    This diagnostic map does not predict trajectories of moving Humans.
    """
    from tarsphere.geometry import HUMAN_BODY_RADIUS_M
    from tarsphere.infra.perception.online_mapping import OnlineOccupancyGrid

    return OnlineOccupancyGrid.from_official_occupancy(
        load_episode_occupancy(episode).occupancy_path,
        static_obstacles=tuple(
            (
                float(h.start_position[0]),
                float(h.start_position[1]),
                HUMAN_BODY_RADIUS_M,
            )
            for h in episode.humans
            if h.static
        ),
    )


__all__ = [
    "load_episode_occupancy",
    "load_episode_navigation_grid",
    "official_runtime_navigation_mask",
]
