"""Exclusion lock for TARSphere processes that instantiate Isaac.

The exclusion unit is a *runtime slot*, not the user.  Kit's ``SimulationApp``
is a per-process singleton, so parallel evaluation runs one process per slot;
what must never happen is two processes silently claiming the same slot and
exhausting its GPU.  A slot is therefore identified by the GPU that Kit will
render and simulate on (``TARSPHERE_KIT_ACTIVE_GPU``) plus an index within that GPU
(``TARSPHERE_ISAAC_RUNTIME_SLOT``).

On an 8-GPU host the eight workers pin eight different GPUs and coexist with
no further configuration.  Placing a second runtime on a GPU that already has
one requires naming a different slot, which keeps that decision deliberate and
attributable to whoever measured that the GPU has the headroom for it.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
import time
from pathlib import Path
from typing import TextIO

_ACTIVE_GPU_ENV = "TARSPHERE_KIT_ACTIVE_GPU"
_RUNTIME_SLOT_ENV = "TARSPHERE_ISAAC_RUNTIME_SLOT"

# Cold scene loads have a much larger transient RAM / filesystem footprint
# than resident rollouts. These are admission limits, not simulation settings.
_COLD_START_SLOTS = 2
_COLD_START_MIN_MEMORY_GIB = 48
_COLD_START_RESERVED_MEMORY_GIB = 32


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Linux MemAvailable is required for Isaac cold-start admission.")


def acquire_isaac_startup_capacity(*, lock_directory: Path | None = None) -> TextIO:
    """Wait before heavy imports; hold capacity until the first reset finishes.

    A short admission mutex makes the available-memory check and reservation
    atomic. Busy cold starts reserve additional headroom because they may not
    have allocated their peak memory yet. Resident workers hold no token.
    """
    if lock_directory is None:
        from tarsphere.infra.process_lifecycle import host_state_directory

        lock_directory = host_state_directory() / "isaac_startup"
    lock_directory.mkdir(parents=True, exist_ok=True)
    while True:
        free: list[TextIO] = []
        busy = 0
        with (lock_directory / "admission.lock").open("a") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            try:
                for index in range(_COLD_START_SLOTS):
                    token = (lock_directory / f"cold{index}.lock").open("a")
                    try:
                        fcntl.flock(token, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        token.close()
                        busy += 1
                    else:
                        free.append(token)
                available = available_memory_gib()
                required = (
                    _COLD_START_MIN_MEMORY_GIB + busy * _COLD_START_RESERVED_MEMORY_GIB
                )
                if free and available >= required:
                    token = free.pop(0)
                    print(
                        f"[WORKER] Cold-start admitted: available={available:.1f} GiB, "
                        f"other_cold_starts={busy}.",
                        flush=True,
                    )
                    return token
            finally:
                for token in free:
                    token.close()
        print(
            f"[WORKER] Waiting for cold-start capacity: available={available:.1f} GiB, "
            f"required={required} GiB, cold_starts={busy}/{_COLD_START_SLOTS}.",
            flush=True,
        )
        time.sleep(20)


def isaac_runtime_slot() -> str:
    """Identity of the runtime slot this process claims.

    ``TARSPHERE_KIT_ACTIVE_GPU`` unset means Kit picks its own device, which cannot
    be distinguished per process, so every such process shares one slot.
    """

    gpu = os.environ.get(_ACTIVE_GPU_ENV, "").strip() or "default"
    slot = os.environ.get(_RUNTIME_SLOT_ENV, "").strip() or "0"
    for name, value in ((_ACTIVE_GPU_ENV, gpu), (_RUNTIME_SLOT_ENV, slot)):
        if value != "default" and not value.isdigit():
            raise RuntimeError(f"{name} must be a non-negative integer, got {value!r}.")
    return f"gpu{gpu}_slot{slot}"


def acquire_isaac_process_lock(
    owner: str,
    *,
    lock_directory: Path | None = None,
) -> TextIO:
    """Hold one Isaac runtime slot until the returned file closes."""

    owner = str(owner).strip()
    if not owner:
        raise ValueError("Isaac process-lock owner must be non-empty.")
    slot = isaac_runtime_slot()
    lock_directory = lock_directory or Path(tempfile.gettempdir())
    lock_path = lock_directory / f"tarsphere_isaac_{os.getuid()}_{slot}.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        active_owner = lock_file.read().strip() or "unknown owner"
        lock_file.close()
        raise RuntimeError(
            f"Isaac runtime slot {slot} is already owned by {active_owner}; "
            f"refusing concurrent startup for {owner}. Pin a different GPU "
            f"with {_ACTIVE_GPU_ENV}, or set {_RUNTIME_SLOT_ENV} to place a "
            "second runtime on this GPU once its memory headroom is known."
        ) from exc
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()} owner={owner} slot={slot}")
    lock_file.flush()
    return lock_file
