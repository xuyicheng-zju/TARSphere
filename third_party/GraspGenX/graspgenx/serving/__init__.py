# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ-based inference serving layer for GraspGenX.

Exposes a REQ/REP wire protocol (msgpack + msgpack-numpy) so that lightweight
clients — including the MCP bridge under ``mcp/`` and the CLI under
``client-server/`` — can drive grasp inference without loading any model
weights themselves.
"""

from graspgenx.serving.types import SweepVolumeParams
from graspgenx.serving.zmq_client import GraspGenXClient
from graspgenx.serving.zmq_server import GraspGenXZMQServer

__all__ = ["GraspGenXClient", "GraspGenXZMQServer", "SweepVolumeParams"]
