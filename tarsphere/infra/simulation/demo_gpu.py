"""Strict physical identity for explicitly isolated demo CUDA processes.

Kit's renderer and resource locks retain the physical GPU index. CUDA consumers
see one full UUID and therefore use ordinal zero. No CUDA/Isaac import occurs
here; nvidia-smi reads the host's NVML physical index-to-UUID mapping.
"""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
from typing import Mapping
import uuid


def _canonical_gpu_uuid(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("GPU-"):
        raise RuntimeError("Demo isolation requires a canonical full NVIDIA GPU UUID.")
    try:
        canonical = "GPU-" + str(uuid.UUID(value[4:]))
    except ValueError as exc:
        raise RuntimeError(
            "Demo isolation requires a canonical full NVIDIA GPU UUID."
        ) from exc
    if value != canonical:
        raise RuntimeError("Demo isolation requires a canonical full NVIDIA GPU UUID.")
    return canonical


def lookup_physical_gpu_uuid(physical_gpu: int) -> str:
    """Return the exact physical NVML device identity, or fail before startup."""
    if (
        isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
        or physical_gpu < 0
    ):
        raise RuntimeError("Physical GPU index must be a non-negative integer.")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_gpu}",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Cannot verify NVML identity for physical GPU {physical_gpu}."
        ) from exc
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError(
            f"NVML did not return exactly one identity for physical GPU {physical_gpu}."
        )
    raw_index, raw_uuid = (item.strip() for item in rows[0])
    if raw_index != str(physical_gpu):
        raise RuntimeError(
            f"NVML GPU index mismatch: requested {physical_gpu}, received {raw_index!r}."
        )
    return _canonical_gpu_uuid(raw_uuid)


def resolve_demo_gpu_mapping(environ: Mapping[str, str] | None = None) -> dict | None:
    """Validate the explicit demo mapping; return None for the normal mode.

    An arbitrary CUDA mask never activates this mode. Callers retain their
    existing unmasked contracts when TARSPHERE_DEMO_GPU_UUID is unset/empty.
    The result's physical_gpu is the host NVML/Kit index used for locks;
    uuid is that device's full identity, and cuda_ordinal is zero within the
    process's single-device CUDA view.
    """
    environment = os.environ if environ is None else environ
    requested = environment.get("TARSPHERE_DEMO_GPU_UUID", "")
    if requested == "":
        return None
    canonical = _canonical_gpu_uuid(requested)
    if environment.get("CUDA_VISIBLE_DEVICES") != canonical:
        raise RuntimeError(
            "Demo CUDA_VISIBLE_DEVICES must contain only the exact TARSPHERE_DEMO_GPU_UUID."
        )
    physical_text = environment.get("TARSPHERE_KIT_ACTIVE_GPU", "")
    if not re.fullmatch(r"0|[1-9][0-9]*", physical_text):
        raise RuntimeError(
            "Demo TARSPHERE_KIT_ACTIVE_GPU must be a canonical non-negative physical index."
        )
    physical_gpu = int(physical_text)
    observed = lookup_physical_gpu_uuid(physical_gpu)
    if observed != canonical:
        raise RuntimeError(
            f"Demo GPU identity mismatch: physical GPU {physical_gpu} is {observed}, "
            f"but TARSPHERE_DEMO_GPU_UUID is {canonical}."
        )
    return {"physical_gpu": physical_gpu, "uuid": canonical, "cuda_ordinal": 0}


__all__ = ["lookup_physical_gpu_uuid", "resolve_demo_gpu_mapping"]
