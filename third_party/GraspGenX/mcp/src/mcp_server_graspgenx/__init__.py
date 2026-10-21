# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .server import serve


def main():
    """MCP GraspGenX Server — cross-embodiment 6-DOF grasp generation for LLM tool-calling."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="MCP server that bridges LLM tool-calling to a GraspGenX ZMQ inference server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="GraspGenX ZMQ server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="GraspGenX ZMQ server port (default: 5556)",
    )
    args = parser.parse_args()
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
