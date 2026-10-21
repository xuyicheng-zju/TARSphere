"""CUDA allocator setup for the Isaac and perception model stack."""

from __future__ import annotations

import os


def configure_cuda_allocator() -> None:
    """Set the PyTorch CUDA allocator config *before* torch is imported.

    Idempotent and non-destructive: ``setdefault`` keeps any value the user
    already exported. Must be called at the top of an entry script, before
    ``import torch`` and before any Isaac/Kit import — once PyTorch initialises
    CUDA, changing this env var has no effect.

    ``expandable_segments:True`` lets the caching allocator extend segments
    instead of allocating fixed 2 MiB blocks, which is what fragments VRAM when
    the Kit/Isaac runtime and torch (for SAM3) allocate interleaved.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


__all__ = ["configure_cuda_allocator"]
