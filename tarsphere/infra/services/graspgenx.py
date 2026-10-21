#!/usr/bin/env python3
"""Run the pinned GraspGenX service used by TARSphere grasp skills."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
import sys
from typing import Any


# Only inference requests consume random state. Health and metadata calls must
# remain observational and must not perturb the next grasp proposal sequence.
GRASPGENX_INFERENCE_ACTIONS = frozenset(
    {
        "infer",
        "infer_object",
        "infer_scene_depth",
        "infer_scene_pc",
    }
)


def graspgenx_source_root() -> Path:
    """Return the one repository-local GraspGenX source package."""

    source_root = Path(__file__).resolve().parents[3] / "third_party" / "GraspGenX"
    if not (source_root / "graspgenx" / "serving" / "zmq_server.py").is_file():
        raise FileNotFoundError(
            "Bundled GraspGenX source is missing from third_party/GraspGenX. "
            "Restore the complete TARSphere release source tree."
        )
    return source_root


def seed_inference_request(request: dict[str, Any]) -> int:
    """Make one GraspGenX inference request reproducible across episodes."""

    seed = int(request.get("seed", 0))
    if seed < 0:
        raise ValueError("GraspGenX inference seed must be non-negative.")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def argument_parser() -> argparse.ArgumentParser:
    """Describe the TARSPHERE-owned GraspGenX service boundary."""

    source_root = graspgenx_source_root()
    parser = argparse.ArgumentParser(
        description="Run the pinned repository-local GraspGenX service.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=source_root / "ext" / "graspgenx_checkpoints" / "release",
        help=(
            "Directory containing the GraspGenX gen/ and dis/ checkpoints. "
            "This large local asset is not stored in Git."
        ),
    )
    parser.add_argument(
        "--gripper-assets",
        type=Path,
        default=(
            source_root
            / "ext"
            / "gripper_descriptions"
            / "gripper_descriptions"
            / "assets"
        ),
        help="Directory containing the native gripper descriptions.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface on which the local ZMQ service listens.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="TCP port on which the local ZMQ service listens.",
    )
    return parser


def main() -> None:
    """Load the pinned upstream server and add request-scoped determinism."""

    args = argument_parser().parse_args()
    source_root = graspgenx_source_root()
    sys.path.insert(0, str(source_root))

    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    class TARSphereGraspGenXServer(GraspGenXZMQServer):
        """GraspGenX with TARSPHERE's deterministic request contract."""

        def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
            if request.get("action") in GRASPGENX_INFERENCE_ACTIONS:
                seed_inference_request(request)
            return super()._dispatch(request)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    server = TARSphereGraspGenXServer(
        config_path=str(args.checkpoint_root.resolve()),
        assets_dir=str(args.gripper_assets.resolve()),
        host=str(args.host),
        port=int(args.port),
        default_gripper="unitree_g1",
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
