"""Host-local ownership for evaluation processes; never signal an unowned PID."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
import time


def host_state_directory() -> Path:
    host = socket.gethostname().split(".", 1)[0]
    return Path.home() / ".cache" / "tarsphere" / host / "evaluation"


def control_directory(output: Path) -> Path:
    """Keep supervisor files outside the immutable-plan result directory."""
    return output.parent / "evaluation_control" / output.name


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluator owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def process_birth(pid: int) -> str | None:
    """Linux start ticks disambiguate a reused PID; zombies are already exited."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] == "Z":
        return None
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return f"{boot}:{fields[19]}"


def terminate_owned_group(pid: int, birth: str, *, grace_seconds: float = 30) -> None:
    """Stop only the exact recorded session leader and its subprocess group."""
    if process_birth(pid) != birth:
        return
    try:
        if os.getpgid(pid) != pid:
            raise RuntimeError(f"Refusing to signal non-leader PID {pid}")
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while process_birth(pid) == birth and time.monotonic() < deadline:
            time.sleep(0.1)
        if process_birth(pid) == birth:
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def require_parent_lifetime(variable: str) -> None:
    """Linux workers die if their owning scheduler/supervisor disappears."""
    expected = os.environ.get(variable)
    if expected is None:
        return
    if sys.platform != "linux":
        raise RuntimeError("Supervised formal evaluation requires Linux")
    parent = int(expected)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot bind process to parent lifetime")
    # Covers the race where the parent exited before prctl was installed.
    if os.getppid() != parent:
        raise RuntimeError(f"Owner process {parent} exited before child startup")
