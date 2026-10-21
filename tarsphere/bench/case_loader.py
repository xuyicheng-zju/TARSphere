"""Load and validate benchmark case configurations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml

from tarsphere.bench.schema import TARSphereEpisode


class BenchmarkCaseLoader:
    """Load one or more TARSphere-Bench case specifications.

    Case files are validated against the benchmark schema and exposed in a
    deterministic order to runtime entry points.
    """

    def __init__(self, episodes_dir: Optional[str] = None):
        self._cases: List[TARSphereEpisode] = []

        if episodes_dir:
            self.load_directory(episodes_dir)

    def load_directory(self, directory: str) -> int:
        """Atomically append a validated directory; never skip invalid cases."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Episodes directory not found: {directory}")

        episodes = []
        for yaml_file in sorted(dir_path.glob("*.yaml")):
            try:
                episodes.append(self.load_file(yaml_file))
            except (ValueError, yaml.YAMLError) as exc:
                raise ValueError(f"Invalid episode {yaml_file}: {exc}") from exc

        self._cases.extend(episodes)
        return len(episodes)

    def load_file(self, path: Path | str) -> TARSphereEpisode:
        """Load and validate a single episode YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return TARSphereEpisode.model_validate(data or {})

    def add_episode(self, episode: TARSphereEpisode):
        """Add a validated case directly."""
        self._cases.append(episode)

    @property
    def episodes(self) -> List[TARSphereEpisode]:
        """Cases retained under the runtime-facing episode term."""

        return list(self._cases)

    @property
    def count(self) -> int:
        return len(self._cases)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> TARSphereEpisode:
        return self._cases[idx]

    def __iter__(self):
        return iter(self._cases)
