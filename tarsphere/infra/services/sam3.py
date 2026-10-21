#!/usr/bin/env python3
"""Load one SAM3 checkpoint and serve all local evaluation workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tarsphere.harness.perception.sam3_service import (  # noqa: E402
    SAM3ModelState,
    SAM3RequestServer,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=5558)
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="SAM3.1 multiplex checkpoint; defaults to TARSPHERE_SAM3_CHECKPOINT.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    raw_checkpoint = (
        str(args.checkpoint)
        if args.checkpoint is not None
        else os.environ.get("TARSPHERE_SAM3_CHECKPOINT", "")
    )
    checkpoint = Path(raw_checkpoint).expanduser()
    if not raw_checkpoint or not checkpoint.is_file():
        raise SystemExit(
            "SAM3 checkpoint is missing. Pass --checkpoint or set "
            "TARSPHERE_SAM3_CHECKPOINT."
        )
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")

    state = SAM3ModelState(str(checkpoint.resolve()))
    print(f"[SAM3] Loading {checkpoint} ...", flush=True)
    state.require_ready()
    server = SAM3RequestServer((args.host, args.port), state)
    print(
        f"[SAM3] Ready on {args.host}:{args.port}; pid={os.getpid()}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
