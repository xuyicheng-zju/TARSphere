#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List grippers available to GraspGenX.

By default this enumerates the real-world ``x_grippers`` shipped with the
``gripper_descriptions`` repo (auto-resolved through ``graspgenx``'s setup
hook — no separate ``pip install`` required). If a local
``assets/proc_grippers/`` directory exists alongside the repo (used by some
training/eval workflows), its contents are listed too.

Usage:
    python scripts/list_grippers.py
    python scripts/list_grippers.py --source x_grippers
    python scripts/list_grippers.py --source proc_grippers
    python scripts/list_grippers.py --filter parallel
    python scripts/list_grippers.py --paths        # show absolute directories
    python scripts/list_grippers.py --json         # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Triggers the on-import hook that clones gripper_descriptions if missing and
# registers its location on sys.path so ``import gripper_descriptions`` works.
import graspgenx  # noqa: F401
import gripper_descriptions

PROC_GRIPPERS_PATH = Path("assets/proc_grippers")


def _discover_x_grippers(name_filter: Optional[str]) -> List[Tuple[str, Path]]:
    assets_root = Path(gripper_descriptions.get_assets_path())
    grippers: List[Tuple[str, Path]] = []
    for name in gripper_descriptions.list_grippers():
        if name_filter and name_filter not in name:
            continue
        grippers.append((name, assets_root / name))
    return grippers


def _discover_proc_grippers(name_filter: Optional[str]) -> List[Tuple[str, Path]]:
    if not PROC_GRIPPERS_PATH.is_dir():
        return []
    grippers: List[Tuple[str, Path]] = []
    for entry in sorted(PROC_GRIPPERS_PATH.iterdir()):
        if not entry.is_dir():
            continue
        # Match the validation in vis_all_grippers.py
        if not (entry / "gripper.urdf").exists():
            continue
        if not (entry / "config.json").exists():
            continue
        if name_filter and name_filter not in entry.name:
            continue
        grippers.append((entry.name, entry.resolve()))
    return grippers


def discover(
    source: str, name_filter: Optional[str]
) -> Dict[str, List[Tuple[str, Path]]]:
    result: Dict[str, List[Tuple[str, Path]]] = {}
    if source in ("both", "x_grippers"):
        result["x_grippers"] = _discover_x_grippers(name_filter)
    if source in ("both", "proc_grippers"):
        result["proc_grippers"] = _discover_proc_grippers(name_filter)
    return result


def _print_human(groups: Dict[str, List[Tuple[str, Path]]], show_paths: bool) -> None:
    total = sum(len(v) for v in groups.values())
    print(f"Found {total} gripper(s).")
    for source, entries in groups.items():
        if not entries:
            continue
        print(f"\n--- {source} ({len(entries)}) ---")
        for i, (name, path) in enumerate(entries, start=1):
            if show_paths:
                print(f"  {i:2d}. {name}  [{path}]")
            else:
                print(f"  {i:2d}. {name}")
    if total == 0:
        print(
            "\n(No grippers discovered. Check that gripper_descriptions cloned "
            "successfully or that $GRASPGENX_GRIPPER_CFG_DIR is set.)"
        )


def _print_json(groups: Dict[str, List[Tuple[str, Path]]]) -> None:
    payload = {
        source: [{"name": name, "path": str(path)} for name, path in entries]
        for source, entries in groups.items()
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write(os.linesep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List grippers available to GraspGenX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["both", "x_grippers", "proc_grippers"],
        default="both",
        help="Which gripper set(s) to enumerate (default: both).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only include grippers whose name contains this substring.",
    )
    parser.add_argument(
        "--paths",
        action="store_true",
        help="Show the absolute asset directory next to each gripper name.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON (overrides --paths formatting).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = discover(args.source, args.filter)
    if args.json:
        _print_json(groups)
    else:
        _print_human(groups, show_paths=args.paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
