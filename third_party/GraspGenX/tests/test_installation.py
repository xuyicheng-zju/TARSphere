"""Smoke tests verifying the package installs correctly and core imports work."""

import importlib
import sys


def test_graspgenx_importable():
    import graspgenx
    assert hasattr(graspgenx, "__version__")
    assert graspgenx.__version__ == "0.1.0"


def test_core_submodules_importable():
    """Verify all core submodules can be imported."""
    modules = [
        "graspgenx",
        "graspgenx.utils.logging_config",
        "graspgenx.utils.compute_utils",
        "graspgenx.utils.point_cloud",
        "graspgenx.robot",
        "graspgenx.dataset.exceptions",
        "graspgenx.dataset.webdataset_utils",
    ]
    for module_name in modules:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


def test_version_consistency():
    """Check version in __init__.py is a valid semver-ish string."""
    import graspgenx
    parts = graspgenx.__version__.split(".")
    assert len(parts) >= 2, f"Version {graspgenx.__version__} doesn't look like semver"
    for part in parts:
        assert part.isdigit(), f"Non-numeric version component: {part}"


def test_logging_configured_on_import():
    """Logging should be configured when graspgenx is imported."""
    import logging
    from graspgenx.utils.logging_config import get_logger, _logging_initialized
    logger = get_logger("test_install_check")
    assert isinstance(logger, logging.Logger)
    assert _logging_initialized is True


def test_numpy_and_torch_available():
    """Verify critical dependencies are installed."""
    import numpy as np
    import torch
    assert np.__version__ is not None
    assert torch.__version__ is not None


def test_trimesh_available():
    """Verify trimesh is available."""
    import trimesh
    assert trimesh.__version__ is not None
