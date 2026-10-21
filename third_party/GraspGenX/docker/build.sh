#!/usr/bin/env bash
set -euo pipefail

VER=3.0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building Docker image x_grasp:$VER ==="
cd "$REPO_DIR"
docker build -f docker/Dockerfile --progress=plain . --network=host -t x_grasp:$VER -t x_grasp:latest

echo "=== Done ==="
