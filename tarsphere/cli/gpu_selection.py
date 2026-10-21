"""Resolve CUDA visibility to physical GPU indices used by Isaac Kit."""

from __future__ import annotations

import os
import subprocess


def select_gpu_indices(
    devices: list[tuple[int, str]], visibility: str | None
) -> list[int]:
    """Honor numeric indices and unique CUDA GPU UUID prefixes."""
    if visibility is None:
        return [index for index, _ in devices]
    if not visibility.strip() or visibility.strip() == "-1":
        return []
    selected: list[int] = []
    for token in visibility.split(","):
        token = token.strip()
        matches = [
            index
            for index, uuid in devices
            if token == str(index)
            or (token.startswith("GPU-") and uuid.startswith(token))
        ]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError(
                f"Invalid or duplicate CUDA_VISIBLE_DEVICES entry: {token!r}"
            )
        selected.append(matches[0])
    return selected


def visible_gpu_indices() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=30,
    )
    devices = [
        (int(index.strip()), uuid.strip())
        for index, uuid in (line.split(",", 1) for line in output.splitlines())
    ]
    return select_gpu_indices(devices, os.environ.get("CUDA_VISIBLE_DEVICES"))


if __name__ == "__main__":
    indices = visible_gpu_indices()
    requested = os.environ.get("TARSPHERE_MODEL_SERVICE_GPU")
    if not indices:
        raise SystemExit("CUDA_VISIBLE_DEVICES exposes no GPUs")
    if requested is not None and int(requested) not in indices:
        raise SystemExit("TARSPHERE_MODEL_SERVICE_GPU is outside CUDA_VISIBLE_DEVICES")
    print(requested if requested is not None else indices[0])
