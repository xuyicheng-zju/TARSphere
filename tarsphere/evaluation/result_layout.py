"""One official result per model/scene/task-family/slice; diagnostics elsewhere."""

from __future__ import annotations

from pathlib import Path
import time


def result_directory(
    scene_directory: Path, mainline: str, capability_slice: str
) -> Path:
    for value in (mainline, capability_slice):
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"Invalid result path component: {value!r}")
    return scene_directory / mainline / capability_slice


def prepare_result_directory(directory: Path) -> None:
    """Preserve an incomplete execution outside the scored tree before restarting.

    The scheduler must first validate and exclude completed results, and hold
    the host/output ownership locks. No simulator mid-episode resume is implied.
    """
    if directory.exists() and any(directory.iterdir()):
        root = directory.parents[3]  # root/model/scene/mainline/slice
        diagnostic = (
            root.parent
            / "evaluation_diagnostics"
            / root.name
            / directory.relative_to(root)
            / str(time.time_ns())
        )
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(diagnostic)
    directory.mkdir(parents=True, exist_ok=True)
