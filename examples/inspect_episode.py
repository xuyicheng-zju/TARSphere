"""Inspect an episode without starting the simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from tarsphere.bench.case_loader import BenchmarkCaseLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    episode = BenchmarkCaseLoader().load_file(args.episode)
    profile = episode.evaluation.task_profile
    print(f"Episode: {episode.episode_id}")
    print(f"Task: {profile.mainline.value.replace('_', '-')}")
    print(f"Condition: {profile.capability_slice.value.replace('_', ' ')}")
    print(f"Scene: {episode.scene.asset_path}")
    for milestone in episode.evaluation.milestones:
        print(f"  {milestone.milestone_id}")


if __name__ == "__main__":
    main()
