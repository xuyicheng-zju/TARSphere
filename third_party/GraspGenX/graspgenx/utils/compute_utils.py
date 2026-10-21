# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute-resource diagnostics for CPU, RAM, GPU, disk, and SLURM.

All helpers read from ``/proc`` and ``os.statvfs`` directly so they work
inside any Linux container without extra pip dependencies.

Usage from other modules::

    from graspgenx.utils.compute_utils import log_system_memory, log_disk_space, log_gpu_memory, log_slurm_info

    log_system_memory(logger, tag="after_cache_load")
"""

import os
from logging import Logger

# ── helpers ──────────────────────────────────────────────────────────────


def _parse_proc_meminfo() -> dict:
    """Parse /proc/meminfo → dict of key → value in bytes."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            key = parts[0].rstrip(":")
            val_kb = int(parts[1])
            info[key] = val_kb * 1024
    return info


def _parse_proc_self_status() -> dict:
    """Parse /proc/self/status for Vm* fields → dict (bytes)."""
    info = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Vm") or line.startswith("Rss"):
                parts = line.split()
                key = parts[0].rstrip(":")
                val_kb = int(parts[1])
                info[key] = val_kb * 1024
    return info


def _fmt(nbytes: float) -> str:
    """Human-readable bytes → string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PiB"


# ── public diagnostic functions ──────────────────────────────────────────


def _read_cgroup_memory_limit() -> tuple:
    """Read the cgroup memory limit and current usage.

    Returns (limit_bytes, usage_bytes) or (None, None) if unavailable.
    Tries cgroups v2 first, then falls back to cgroups v1.
    """
    # cgroups v2: /sys/fs/cgroup/memory.max and /sys/fs/cgroup/memory.current
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        limit = None if raw == "max" else int(raw)
        with open("/sys/fs/cgroup/memory.current") as f:
            usage = int(f.read().strip())
        return limit, usage
    except (FileNotFoundError, PermissionError, ValueError):
        pass

    # cgroups v1: try several known paths
    v1_paths = [
        # SLURM step-level cgroup (most specific)
        "/sys/fs/cgroup/memory/slurm",
        # Generic paths
        "/sys/fs/cgroup/memory",
    ]
    for base in v1_paths:
        limit_file = os.path.join(base, "memory.limit_in_bytes")
        usage_file = os.path.join(base, "memory.usage_in_bytes")
        try:
            with open(limit_file) as f:
                limit_raw = int(f.read().strip())
            with open(usage_file) as f:
                usage_raw = int(f.read().strip())
            # A limit of 2^63 - page_size (or similar huge value) means "no limit"
            limit = None if limit_raw >= (1 << 62) else limit_raw
            return limit, usage_raw
        except (FileNotFoundError, PermissionError, ValueError):
            continue

    return None, None


def log_system_memory(logger: Logger, tag: str = "") -> None:
    """Log CPU core count, system-wide RAM, cgroup limit, and per-process memory."""
    prefix = f"[MEM {tag}]" if tag else "[MEM]"

    cpu_count = os.cpu_count()
    logger.info(f"{prefix} CPU logical cores: {cpu_count}")

    try:
        mem = _parse_proc_meminfo()
        logger.info(
            f"{prefix} System RAM total:     {_fmt(mem['MemTotal'])}  (NOTE: /proc/meminfo reports HOST memory, not cgroup limit)"
        )
        logger.info(f"{prefix} System RAM free:      {_fmt(mem['MemFree'])}")
        logger.info(f"{prefix} System RAM available: {_fmt(mem['MemAvailable'])}")
        logger.info(f"{prefix} Buffers:              {_fmt(mem['Buffers'])}")
        logger.info(f"{prefix} Cached:               {_fmt(mem['Cached'])}")
        logger.info(f"{prefix} SwapTotal:            {_fmt(mem['SwapTotal'])}")
        logger.info(f"{prefix} SwapFree:             {_fmt(mem['SwapFree'])}")
        used = mem["MemTotal"] - mem["MemAvailable"]
        pct = 100.0 * used / mem["MemTotal"] if mem["MemTotal"] else 0
        logger.info(f"{prefix} System RAM used:      {_fmt(used)} ({pct:.1f}%)")
    except Exception as e:
        logger.warning(f"{prefix} /proc/meminfo unavailable: {e}")

    # ── cgroup memory limit (the actual enforced limit by SLURM) ──
    try:
        cg_limit, cg_usage = _read_cgroup_memory_limit()
        if cg_limit is not None:
            cg_pct = 100.0 * cg_usage / cg_limit if cg_limit else 0
            logger.info(
                f"{prefix} Cgroup mem limit:     {_fmt(cg_limit)}  ← THIS is the real OOM boundary"
            )
            logger.info(
                f"{prefix} Cgroup mem usage:     {_fmt(cg_usage)} ({cg_pct:.1f}%)"
            )
        elif cg_usage is not None:
            logger.info(f"{prefix} Cgroup mem limit:     unlimited (no cap)")
            logger.info(f"{prefix} Cgroup mem usage:     {_fmt(cg_usage)}")
        else:
            logger.info(
                f"{prefix} Cgroup mem limit:     (not readable — cgroup fs not accessible)"
            )
    except Exception as e:
        logger.warning(f"{prefix} Cgroup memory check failed: {e}")

    try:
        vm = _parse_proc_self_status()
        logger.info(f"{prefix} Process VmSize (virtual):  {_fmt(vm.get('VmSize', 0))}")
        logger.info(f"{prefix} Process VmRSS  (resident): {_fmt(vm.get('VmRSS', 0))}")
        logger.info(f"{prefix} Process VmPeak (peak):     {_fmt(vm.get('VmPeak', 0))}")
        logger.info(f"{prefix} Process VmSwap (swap):     {_fmt(vm.get('VmSwap', 0))}")
    except Exception as e:
        logger.warning(f"{prefix} /proc/self/status unavailable: {e}")


def log_gpu_memory(logger: Logger, tag: str = "") -> None:
    """Log per-GPU memory via torch.cuda (only safe after CUDA init)."""
    prefix = f"[GPU {tag}]" if tag else "[GPU]"
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info(f"{prefix} CUDA not available")
            return
        n = torch.cuda.device_count()
        logger.info(f"{prefix} CUDA devices visible: {n}")
        for i in range(n):
            free, total = torch.cuda.mem_get_info(i)
            name = torch.cuda.get_device_name(i)
            used = total - free
            pct = 100.0 * used / total if total else 0
            logger.info(
                f"{prefix}   GPU {i} ({name}): "
                f"{_fmt(used)} / {_fmt(total)} used ({pct:.1f}%),  "
                f"{_fmt(free)} free"
            )
    except Exception as e:
        logger.warning(f"{prefix} torch.cuda error: {e}")


def log_disk_space(logger: Logger, tag: str = "") -> None:
    """Log disk space for key ORD local-NVMe paths.

    Note: previously iterated /proc/mounts and called os.statvfs() on each, but
    that hangs indefinitely on stale lustre/NFS mounts inside pyxis containers,
    causing silent SIGKILL ~60s into the job before training starts.
    """
    prefix = f"[DISK {tag}]" if tag else "[DISK]"

    # Key ORD node local-disk paths
    logger.info(f"{prefix} --- Key paths (ORD node local NVMe) ---")
    key_paths = ["/tmp", "/raid", "/raid/scratch", "/dev/shm"]
    for path in key_paths:
        if os.path.exists(path):
            try:
                st = os.statvfs(path)
                total = st.f_blocks * st.f_frsize
                free = st.f_bavail * st.f_frsize
                used = total - (st.f_bfree * st.f_frsize)
                pct = 100.0 * used / total if total else 0
                logger.info(
                    f"{prefix}   {path:<30} total={_fmt(total):>10}  "
                    f"used={_fmt(used):>10}  avail={_fmt(free):>10}  ({pct:.1f}% used)"
                )
            except (OSError, PermissionError):
                logger.info(f"{prefix}   {path:<30} (exists but stat failed)")
        else:
            logger.info(f"{prefix}   {path:<30} (not found)")


def log_slurm_info(logger: Logger, tag: str = "") -> None:
    """Log relevant SLURM environment variables."""
    prefix = f"[SLURM {tag}]" if tag else "[SLURM]"
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS",
        "SLURM_GPUS_PER_NODE",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
    ]
    found_any = False
    for var in slurm_vars:
        val = os.environ.get(var)
        if val is not None:
            logger.info(f"{prefix} {var} = {val}")
            found_any = True
    if not found_any:
        logger.info(
            f"{prefix} No SLURM variables found (not running inside a SLURM job)"
        )


def log_all_resources(logger: Logger, tag: str = "", include_gpu: bool = False) -> None:
    """Convenience: log SLURM info, system memory, disk, and optionally GPU.

    Parameters
    ----------
    logger : Logger
        Python logger instance.
    tag : str
        Label for this checkpoint (e.g. ``"startup"``, ``"after_cache_load"``).
    include_gpu : bool
        Whether to query GPU memory.  Set to ``False`` before CUDA init /
        before fork to avoid initialising the CUDA runtime.
    """
    logger.info(f"{'='*70}")
    logger.info(f"  RESOURCE DIAGNOSTICS  —  checkpoint: {tag or 'N/A'}")
    logger.info(f"{'='*70}")
    log_slurm_info(logger, tag)
    log_system_memory(logger, tag)
    log_disk_space(logger, tag)
    if include_gpu:
        log_gpu_memory(logger, tag)
    logger.info(f"{'='*70}")
