# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from graspgenx.utils.logging_config import setup_logging

    setup_logging()
except ImportError:
    pass

try:
    from graspgenx._setup_dependencies import (
        ensure_checkpoints,
        ensure_gripper_descriptions,
        get_checkpoints_root,
        get_checkpoints_version_dir,
        get_gripper_descriptions_assets,
        get_gripper_descriptions_root,
    )

    ensure_gripper_descriptions()
    ensure_checkpoints()
except Exception:  # noqa: BLE001 - never let setup hook break imports
    import logging

    logging.getLogger(__name__).debug(
        "graspgenx dependency setup hook failed; continuing.", exc_info=True
    )

__version__ = "0.1.0"
