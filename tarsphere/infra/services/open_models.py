#!/usr/bin/env python3
"""Start or verify user-configured OpenAI-compatible model endpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Service:
    name: str
    model: str
    cuda_visible_devices: str
    executable: str
    port: int
    argv: list[str]


def expand(value: str, *, field: str) -> str:
    result = os.path.expanduser(os.path.expandvars(value))
    if "$" in result or not result.strip():
        raise ValueError(f"{field} is unresolved: {value}")
    return result


def load_services(path: Path) -> list[Service]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "tarsphere.open_model_services.v1":
        raise ValueError(
            f"Unsupported model-service schema: {raw.get('schema_version')!r}"
        )
    services = [
        Service(
            name=str(item["name"]),
            model=str(item["model"]),
            cuda_visible_devices=str(item["cuda_visible_devices"]),
            executable=expand(
                str(item["executable"]), field=f"{item['name']}.executable"
            ),
            port=int(item["port"]),
            argv=[
                expand(str(value), field=f"{item['name']}.argv")
                for value in item["argv"]
            ],
        )
        for item in raw["services"]
    ]
    if not services or len({item.name for item in services}) != len(services):
        raise ValueError("Define at least one service with unique names")
    occupied_gpus: set[int] = set()
    for service in services:
        if not Path(service.executable).is_file():
            raise FileNotFoundError(
                f"{service.name} executable missing: {service.executable}"
            )
        if not 1 <= service.port <= 65535:
            raise ValueError(f"{service.name} port must be in 1..65535")
        gpus = {int(value) for value in service.cuda_visible_devices.split(",")}
        overlap = occupied_gpus.intersection(gpus)
        if overlap:
            raise ValueError(f"{service.name} reuses GPU(s) {sorted(overlap)}")
        occupied_gpus.update(gpus)
    return services


def health(service: Service) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{service.port}/v1/models", timeout=5
        ) as response:
            payload: dict[str, Any] = json.loads(response.read())
    except Exception:  # noqa: BLE001 - polling returns a boolean
        return False
    ids = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    return response.status == 200 and service.model in ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.home() / ".cache" / "tarsphere" / "open_models",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    services = load_services(args.config.resolve())

    if args.dry_run:
        for service in services:
            print(
                json.dumps(
                    {
                        "name": service.name,
                        "model": service.model,
                        "gpus": service.cuda_visible_devices,
                        "endpoint": f"http://127.0.0.1:{service.port}/v1",
                        "command": [service.executable, *service.argv],
                    }
                )
            )
        return

    if args.check_only:
        unhealthy = [service.name for service in services if not health(service)]
        if unhealthy:
            raise SystemExit(f"Unhealthy open-model services: {unhealthy}")
        print("[OPEN MODELS] All four endpoints are healthy.")
        return

    args.state_dir.mkdir(parents=True, exist_ok=True)
    started: dict[str, subprocess.Popen[Any]] = {}
    logs: list[Any] = []
    for service in services:
        if health(service):
            print(f"[OPEN MODELS] Reusing {service.name} on port {service.port}.")
            continue
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = service.cuda_visible_devices
        log_path = args.state_dir / f"{service.name}.log"
        log = log_path.open("a", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen(
            [service.executable, *service.argv],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (args.state_dir / f"{service.name}.pid").write_text(f"{process.pid}\n")
        started[service.name] = process
        print(
            f"[OPEN MODELS] Starting {service.name} on GPU(s) "
            f"{service.cuda_visible_devices}; pid={process.pid}."
        )

    deadline = time.monotonic() + args.startup_timeout_seconds
    pending = {service.name: service for service in services if not health(service)}
    while pending and time.monotonic() < deadline:
        for name, service in list(pending.items()):
            process = started.get(name)
            if process is not None and process.poll() is not None:
                raise SystemExit(
                    f"{name} exited with code {process.returncode}; inspect "
                    f"{args.state_dir / (name + '.log')}"
                )
            if health(service):
                print(f"[OPEN MODELS] Ready: {name} on port {service.port}.")
                del pending[name]
        if pending:
            time.sleep(10)
    for log in logs:
        log.close()
    if pending:
        raise SystemExit(f"Model startup timed out: {sorted(pending)}")


if __name__ == "__main__":
    main()
