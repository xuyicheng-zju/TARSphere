"""Load user-supplied episode YAML files or a path index for runtime tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_EPISODES = (
    Path(__file__).resolve().parents[2] / "examples/episodes/navigation.yaml"
)


def load_episode_inputs(
    source: Path, episode_ids: list[str] | None = None
) -> list[tuple[Path, Any]]:
    """Read YAML, a directory of YAMLs, or an index with paths relative to itself."""
    from tarsphere.bench.case_loader import BenchmarkCaseLoader

    source = source.expanduser().resolve()
    indexed_ids: dict[Path, str] = {}
    if source.is_dir():
        paths = sorted([*source.rglob("*.yaml"), *source.rglob("*.yml")])
    elif source.suffix.lower() == ".json":
        index = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(index, dict) or not isinstance(index.get("episodes"), list):
            raise ValueError(f"{source}: expected an object with an episodes list")
        paths = []
        for entry in index["episodes"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"{source}: each episode entry requires a path string")
            path = (source.parent / entry["path"]).resolve()
            paths.append(path)
            if "episode_id" in entry:
                indexed_ids[path] = str(entry["episode_id"])
    elif source.suffix.lower() in {".yaml", ".yml"}:
        paths = [source]
    else:
        raise ValueError(f"Expected episode YAML, directory, or JSON index: {source}")
    if not paths:
        raise ValueError(f"No episode YAML files were found in {source}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{source}: duplicate episode paths")
    loader = BenchmarkCaseLoader()
    episodes = [(path, loader.load_file(path)) for path in paths]
    seen: set[str] = set()
    for path, episode in episodes:
        if episode.episode_id in seen:
            raise ValueError(f"Duplicate episode ID: {episode.episode_id}")
        seen.add(episode.episode_id)
        if path in indexed_ids and indexed_ids[path] != episode.episode_id:
            raise ValueError(f"{path}: index/YAML episode ID mismatch")
    if episode_ids:
        requested = set(episode_ids)
        unknown = requested - seen
        if unknown:
            raise ValueError(f"Unknown episode IDs: {sorted(unknown)}")
        episodes = [
            (path, episode)
            for path, episode in episodes
            if episode.episode_id in requested
        ]
    return episodes
