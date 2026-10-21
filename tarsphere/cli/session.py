#!/usr/bin/env python3
"""Server-side tmux ownership and persistent scheduler recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tarsphere.infra.process_lifecycle import (
    control_directory,
    exclusive_lock,
    host_state_directory,
    process_birth,
    terminate_owned_group,
)
from tarsphere.cli.evaluate import atomic_json, build_parser

RESTART_INITIAL_SECONDS = 60
RESTART_CAP_SECONDS = 900
SCHEDULER_STALL_SECONDS = 7200


def session_name(output: Path, phase: str = "physical") -> str:
    """Return the single canonical session name used on every simulator host."""

    del output
    return "eval" if phase == "physical" else "eval-score"


def completed_count(control: Path, phase: str = "physical") -> int:
    try:
        if phase == "scoring":
            status = json.loads((control.parent / "offline_status.json").read_text())
            return int(status["counts"]["evaluated"])
        status = json.loads((control / "status.json").read_text())
        return int(status["completed_physical_episode_run_count"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def supervise(output: Path, arguments: list[str], phase: str = "physical") -> int:
    """Retry infrastructure exits until completion or an explicit stop."""
    control = control_directory(output) / "host0"
    if phase == "scoring":
        control /= "scoring"
    control.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    failures = 0
    launches = 0
    process = None
    birth = None
    receipt = control / "scheduler.json"
    with exclusive_lock(host_state_directory() / f"{phase}_supervisor.lock"):
        # A killed tmux pane/supervisor cannot leave a second scheduler behind.
        if receipt.is_file():
            previous = json.loads(receipt.read_text())
            terminate_owned_group(previous["pid"], previous["birth"])
        progress = completed_count(control, phase)

        def publish(state: str, **extra) -> None:
            atomic_json(
                control / "session_status.json",
                dict(
                    state=state,
                    updated_at_unix=time.time(),
                    supervisor_pid=os.getpid(),
                    launches=launches,
                    consecutive_no_progress_failures=failures,
                    phase=phase,
                    completed_episode_run_count=completed_count(control, phase),
                    **extra,
                ),
            )

        try:
            while not stop_requested:
                launches += 1
                environment = dict(
                    os.environ,
                    TARSPHERE_EVALUATION_SUPERVISED="1",
                    TARSPHERE_SUPERVISOR_PID=str(os.getpid()),
                )
                command = [
                    "bash",
                    str(ROOT / "scripts/evaluate.sh"),
                    "--foreground",
                    *arguments,
                ]
                if phase == "scoring":
                    command.append("--score")
                with (control / f"scheduler_{time.time_ns()}.log").open("x") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    birth = process_birth(process.pid)
                    last_scheduler_tick = time.monotonic()
                    status_path = control / "status.json"
                    status_mtime = (
                        status_path.stat().st_mtime_ns if status_path.exists() else 0
                    )
                    stalled = False
                    if birth is not None:
                        atomic_json(receipt, dict(pid=process.pid, birth=birth))
                    while process.poll() is None and not stop_requested:
                        if phase == "physical":
                            new_mtime = (
                                status_path.stat().st_mtime_ns
                                if status_path.exists()
                                else 0
                            )
                            if new_mtime != status_mtime:
                                status_mtime = new_mtime
                                last_scheduler_tick = time.monotonic()
                            if (
                                time.monotonic() - last_scheduler_tick
                                > SCHEDULER_STALL_SECONDS
                            ):
                                stalled = True
                                publish("scheduler_stalled", scheduler_pid=process.pid)
                                if birth is not None:
                                    terminate_owned_group(
                                        process.pid, birth, grace_seconds=45
                                    )
                                break
                        publish("running", scheduler_pid=process.pid)
                        time.sleep(2)
                    if stop_requested:
                        break
                    code = process.wait()
                    if stalled:
                        code = 75
                if code == 0:
                    publish(f"{phase}_complete", scheduler_exit_code=0)
                    return 0
                if code in (2, 3, 78, 130):
                    publish(
                        "blocked",
                        scheduler_exit_code=code,
                        reason="Configuration/ownership/plan error or explicit interruption; inspect scheduler log.",
                    )
                    return code
                current_progress = completed_count(control, phase)
                failures = 1 if current_progress > progress else failures + 1
                progress = current_progress
                delay = min(
                    RESTART_INITIAL_SECONDS * 2 ** (failures - 1), RESTART_CAP_SECONDS
                )
                publish(
                    "cooldown",
                    scheduler_exit_code=code,
                    retry_at_unix=time.time() + delay,
                )
                deadline = time.monotonic() + delay
                while not stop_requested and time.monotonic() < deadline:
                    time.sleep(min(2, max(0, deadline - time.monotonic())))
        finally:
            if process is not None and birth is not None:
                terminate_owned_group(process.pid, birth, grace_seconds=45)
                process.wait(timeout=10)
            if stop_requested:
                publish(
                    "stopped",
                    reason="Explicit stop; rerun the same launch command to resume.",
                )
    return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "supervise", "status", "stop"))
    parser.add_argument("--phase", choices=("physical", "scoring"), default="physical")
    args, arguments = parser.parse_known_args()
    parsed = build_parser().parse_args(arguments)
    output = parsed.output_dir.resolve()
    if parsed.launch:
        parser.error("--launch is internal; use scripts/evaluate.sh")
    # Canonical explicit arguments are reused by tmux, including on resume.
    control = control_directory(output) / f"host{0}"
    if args.phase == "scoring":
        control /= "scoring"
    name = session_name(output, args.phase)
    if args.mode == "supervise":
        try:
            return supervise(output, arguments, args.phase)
        except Exception as error:
            atomic_json(
                control / "session_status.json",
                dict(
                    state="blocked",
                    phase=args.phase,
                    updated_at_unix=time.time(),
                    reason=f"{type(error).__name__}: {error}",
                ),
            )
            print(f"Supervisor blocked: {error}", file=sys.stderr)
            return 78
    if args.mode == "status":
        for path in (control / "session_status.json", control / "status.json"):
            print(f"{path}:")
            print(path.read_text() if path.is_file() else "not started")
        print(f"tmux attach -t {shlex.quote(name)}")
        return 0
    if shutil.which("tmux") is None:
        parser.error(
            "Install tmux on this server (for example conda install -c conda-forge tmux)."
        )
    active = (
        subprocess.run(
            ["tmux", "has-session", "-t", f"={name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if args.mode == "stop":
        if active:
            # Ctrl-C reaches the supervisor, which cleans up only owned children.
            subprocess.run(
                ["tmux", "send-keys", "-t", f"{name}:0.0", "C-c"], check=True
            )
            print("Stop requested; verify status becomes stopped before resuming.")
        else:
            print("No active evaluation tmux session.")
        return 0
    if active:
        print(f"Already running: {name}; no duplicate evaluator launched.")
        return 0
    control.mkdir(parents=True, exist_ok=True)
    # Re-source host_env.sh inside the new tmux process; credentials are never
    # embedded in tmux arguments, control JSON, or a laptop SSH tunnel.
    command = shlex.join(
        [
            "bash",
            str(ROOT / "scripts/evaluate.sh"),
            "--supervise",
            *arguments,
            *(["--score"] if args.phase == "scoring" else []),
        ]
    )
    command += f" >> {shlex.quote(str(control / 'supervisor.log'))} 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", name, "-c", str(ROOT), command], check=True
    )
    print(f"Server-owned session: {name}\nResults: {output}\nControl/logs: {control}")
    print(f"Attach: tmux attach -t {shlex.quote(name)}; detach with Ctrl-b d.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
