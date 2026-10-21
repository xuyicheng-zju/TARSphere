"""Shared command-line surface for default harness parameters."""

from __future__ import annotations

import argparse
from typing import Any

from tarsphere.harness.harness import (
    CameraView,
    ControllerConfig,
    DEFAULT_VISUAL_HISTORY_FRAMES,
    DYNAMIC_ACTION_HORIZON,
    AgentHarnessConfig,
    MAIN_TEXT_HISTORY_STEPS,
    MAX_VISUAL_HISTORY_FRAMES,
    ProtocolConfig,
    TEXT_HISTORY_ALL,
)


def add_agent_harness_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the supported harness parameters to one entry point."""

    group = parser.add_argument_group("Harness Protocol")
    group.add_argument("--text-history-horizon", default=MAIN_TEXT_HISTORY_STEPS)
    group.add_argument(
        "--visual-history-horizon",
        type=int,
        default=DEFAULT_VISUAL_HISTORY_FRAMES,
        help=(
            "Number of completed RGB frames shown with CURRENT in one 640x640 "
            "chronological mosaic per camera view. The official default is 0, "
            "which sends the native 640x480 current frame only; 1-4 enable "
            "additional temporal context "
            f"(0-{MAX_VISUAL_HISTORY_FRAMES})."
        ),
    )
    group.add_argument(
        "--action-horizon",
        default=DYNAMIC_ACTION_HORIZON,
        help=(
            "Exact executable calls per ordinary policy inference (1-5), or "
            f"{DYNAMIC_ACTION_HORIZON!r} to let the model choose 1-5."
        ),
    )
    group.add_argument(
        "--camera-views",
        default=CameraView.EGO.value,
        help="Comma-separated canonical order: ego,left_wrist,right_wrist.",
    )

    diagnostic = parser.add_argument_group("Controller timing")
    diagnostic.add_argument(
        "--online-mapping-hz",
        type=float,
        default=1.0,
        help=(
            "Private synchronized ego RGB-D mapping samples per simulated "
            "second during locomotion. They do not call the policy or enter "
            "visual history. Use 0 for decision-boundary-only mapping."
        ),
    )


def agent_harness_config_from_args(args: Any) -> AgentHarnessConfig:
    """Build the single typed run configuration from parsed CLI values."""

    raw_text_horizon = str(args.text_history_horizon).strip().lower()
    text_horizon: int | str
    if raw_text_horizon == TEXT_HISTORY_ALL:
        text_horizon = TEXT_HISTORY_ALL
    else:
        try:
            text_horizon = int(raw_text_horizon)
        except ValueError as exc:
            raise ValueError(
                "--text-history-horizon must be a non-negative integer or all"
            ) from exc
    raw_views = [
        item.strip() for item in str(args.camera_views).split(",") if item.strip()
    ]
    raw_action_horizon = str(args.action_horizon).strip().lower()
    action_horizon: int | str
    if raw_action_horizon == DYNAMIC_ACTION_HORIZON:
        action_horizon = DYNAMIC_ACTION_HORIZON
    else:
        try:
            action_horizon = int(raw_action_horizon)
        except ValueError as exc:
            raise ValueError(
                "--action-horizon must be an integer from 1 to 5 or dynamic"
            ) from exc
    return AgentHarnessConfig(
        protocol=ProtocolConfig(
            text_history_horizon=text_horizon,
            visual_history_horizon=int(args.visual_history_horizon),
            action_horizon=action_horizon,
            camera_views=tuple(CameraView(item) for item in raw_views),
        ),
        controller=ControllerConfig(
            online_mapping_hz=float(args.online_mapping_hz),
        ),
    )


__all__ = ["add_agent_harness_arguments", "agent_harness_config_from_args"]
