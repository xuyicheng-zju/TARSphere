# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-import setup hook that locates (or clones, if missing) external
dependencies the first time ``graspgenx`` is imported in a Python process.

Two repositories are managed here:

1. ``gripper_descriptions`` — per-gripper URDFs, meshes, and ``config.json``.
   Resolution order:
     a. ``$GRASPGENX_GRIPPER_CFG_DIR`` if set (must exist on disk; not cloned).
     b. ``<repo_root>/ext/gripper_descriptions`` otherwise (auto-cloned).

2. ``graspgenx_checkpoints`` — versioned generator + discriminator checkpoints.
   Resolution order:
     a. ``$GRASPGENX_CHECKPOINT_DIR`` if set and exists; falls back to (b)
        with a warning if the env-var path is missing.
     b. ``<repo_root>/ext/graspgenx_checkpoints`` otherwise (auto-cloned).
   Inside the resolved root, callers should use the per-version subdir
   selected via :data:`DEFAULT_CHECKPOINT_VERSION`
   (e.g. ``release/{gen,dis}/``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Common paths
# ---------------------------------------------------------------------------

# repo_root == GraspGenX/, i.e. parent of the package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = REPO_ROOT / "ext"

# ---------------------------------------------------------------------------
# gripper_descriptions
# ---------------------------------------------------------------------------

GRIPPER_DESCRIPTIONS_URL = "https://huggingface.co/datasets/adithyamurali/gripper_descriptions"
DEFAULT_GRIPPER_DESCRIPTIONS_DIR = EXT_DIR / "gripper_descriptions"

# Override path entirely via env var (skips auto-clone — the path must exist).
_GRIPPER_DESCRIPTIONS_ENV_VAR = "GRASPGENX_GRIPPER_CFG_DIR"
# Backwards-compatibility alias for older code paths.
_PATH_ENV_VAR = _GRIPPER_DESCRIPTIONS_ENV_VAR

# ---------------------------------------------------------------------------
# graspgenx_checkpoints
# ---------------------------------------------------------------------------

CHECKPOINTS_URL = "https://huggingface.co/adithyamurali/GraspGenXModel"
DEFAULT_CHECKPOINTS_DIR = EXT_DIR / "graspgenx_checkpoints"

# The checkpoint repo is laid out as <root>/<version>/{gen,dis}/. Update this
# constant when a newer release supersedes the current default.
DEFAULT_CHECKPOINT_VERSION = "release"

# If set and existing, redirects checkpoint lookup to the user-supplied path.
# If set but the path is missing, we log a warning and fall back to
# DEFAULT_CHECKPOINTS_DIR (auto-cloning when needed).
_CHECKPOINTS_ENV_VAR = "GRASPGENX_CHECKPOINT_DIR"

# ---------------------------------------------------------------------------
# Setup machinery
# ---------------------------------------------------------------------------

_setup_lock = threading.Lock()
_setup_done_gripper_descriptions = False
_setup_done_checkpoints = False
# Backwards-compatibility alias.
_setup_done = False


def _is_valid_clone(path: Path) -> bool:
    """A directory counts as a valid clone if it contains a ``.git`` entry
    or is otherwise non-empty (e.g., a manually populated checkout)."""
    if not path.exists():
        return False
    if (path / ".git").exists():
        return True
    try:
        return any(path.iterdir())
    except OSError:
        return False


def _git_clone_to(url: str, target: Path, dep_label: str, env_var: str) -> bool:
    """Download ``url`` into ``target`` via ``git clone``. Returns True on success.

    Git's own ``--progress`` output is streamed live to stderr so the user can
    see both object-fetch and LFS-download progress in real time. We do not
    capture stdout/stderr — letting them flow to the terminal naturally is the
    cheapest way to get a usable progress bar without re-implementing one in
    Python.
    """
    if shutil.which("git") is None:
        logger.warning(
            "git executable not found; cannot download %s. "
            "Install git or set %s=<path-to-existing-checkout>.",
            url,
            env_var,
        )
        return False

    try:
        EXT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create %s: %s", EXT_DIR, exc)
        return False

    logger.info(
        "Downloading %s to %s. This may take a few minutes for large LFS "
        "assets (checkpoints can be >1 GB) — progress will be streamed below.",
        dep_label,
        target,
    )
    try:
        result = subprocess.run(
            ["git", "clone", "--progress", "--depth", "1", url, str(target)],
            check=False,
        )
    except OSError as exc:
        logger.warning("Failed to invoke git clone for %s: %s", dep_label, exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "Failed to download %s (git clone exit %s). "
            "Check network connectivity, or set %s=<path-to-existing-checkout>.",
            dep_label,
            result.returncode,
            env_var,
        )
        return False

    logger.info("%s ready at %s", dep_label, target)
    return True


# ---------------------------------------------------------------------------
# gripper_descriptions resolution
# ---------------------------------------------------------------------------


def _register_gripper_descriptions_on_sys_path(root: Path) -> None:
    """Make ``import gripper_descriptions`` work without ``pip install``.

    The ``gripper_descriptions`` repo is laid out as
    ``<root>/gripper_descriptions/__init__.py`` (a standard src-less Python
    package next to its own ``pyproject.toml``). To let users run
    ``python -m gripper_descriptions.scripts.vis_all_grippers`` after a plain
    ``import graspgenx``, we prepend the clone root to :data:`sys.path` so the
    inner package becomes discoverable. The insertion is idempotent.
    """
    if not root.exists():
        return
    pkg_dir = root / "gripper_descriptions"
    if not pkg_dir.is_dir():
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def get_gripper_descriptions_root() -> Path:
    """Return the absolute path to the gripper_descriptions checkout.

    Resolution:
        1. ``$GRASPGENX_GRIPPER_CFG_DIR`` if set (must exist).
        2. ``<repo_root>/ext/gripper_descriptions`` otherwise (may be auto-cloned
           by :func:`ensure_gripper_descriptions`).

    Raises:
        FileNotFoundError: if the env var is set but the path does not exist.
    """
    override = os.environ.get(_GRIPPER_DESCRIPTIONS_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"{_GRIPPER_DESCRIPTIONS_ENV_VAR} points to {path}, which does not exist. "
                f"Either create that directory (clone gripper_descriptions there) "
                f"or unset {_GRIPPER_DESCRIPTIONS_ENV_VAR} to use the default at "
                f"{DEFAULT_GRIPPER_DESCRIPTIONS_DIR}."
            )
        _register_gripper_descriptions_on_sys_path(path)
        return path
    _register_gripper_descriptions_on_sys_path(DEFAULT_GRIPPER_DESCRIPTIONS_DIR)
    return DEFAULT_GRIPPER_DESCRIPTIONS_DIR


def get_gripper_descriptions_assets() -> Path:
    """Path to ``gripper_descriptions/assets/x_grippers/`` inside the checkout.

    This is the directory that contains per-gripper folders (``fetch/``,
    ``franka_panda/``, …) consumed by the wizard and the viewer.
    """
    return (
        get_gripper_descriptions_root()
        / "gripper_descriptions"
        / "assets"
        / "x_grippers"
    )


def ensure_gripper_descriptions(force: bool = False) -> Path | None:
    """Make sure ``gripper_descriptions`` is available locally.

    If ``$GRASPGENX_GRIPPER_CFG_DIR`` is set, this only verifies the directory
    exists and returns its path — no cloning is performed (the user has
    explicitly chosen that location). Otherwise, the default location at
    ``<repo_root>/ext/gripper_descriptions`` is created via ``git clone`` if
    missing.
    """
    global _setup_done_gripper_descriptions, _setup_done

    override = os.environ.get(_GRIPPER_DESCRIPTIONS_ENV_VAR)
    if override:
        try:
            path = get_gripper_descriptions_root()
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            return None
        _register_gripper_descriptions_on_sys_path(path)
        return path

    target = DEFAULT_GRIPPER_DESCRIPTIONS_DIR

    with _setup_lock:
        if _setup_done_gripper_descriptions and not force:
            if _is_valid_clone(target):
                _register_gripper_descriptions_on_sys_path(target)
                return target
            return None

        if _is_valid_clone(target) and not force:
            _setup_done_gripper_descriptions = True
            _setup_done = True
            _register_gripper_descriptions_on_sys_path(target)
            return target

        if force and target.exists():
            logger.info("Removing existing %s for re-clone.", target)
            shutil.rmtree(target, ignore_errors=True)

        if not _git_clone_to(
            GRIPPER_DESCRIPTIONS_URL,
            target,
            dep_label="gripper_descriptions",
            env_var=_GRIPPER_DESCRIPTIONS_ENV_VAR,
        ):
            return None

        _setup_done_gripper_descriptions = True
        _setup_done = True
        _register_gripper_descriptions_on_sys_path(target)
        return target


# ---------------------------------------------------------------------------
# graspgenx_checkpoints resolution
# ---------------------------------------------------------------------------


def get_checkpoints_root() -> Path:
    """Return the absolute path to the graspgenx_checkpoints checkout.

    Resolution:
        1. ``$GRASPGENX_CHECKPOINT_DIR`` if set **and** exists on disk.
        2. ``<repo_root>/ext/graspgenx_checkpoints`` otherwise (may be auto-cloned
           by :func:`ensure_checkpoints`).

    If ``$GRASPGENX_CHECKPOINT_DIR`` is set but the path does not exist, a
    warning is logged and the default location is returned (the auto-clone
    fallback will populate it on demand).
    """
    override = os.environ.get(_CHECKPOINTS_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return path
        logger.warning(
            "%s points to %s, which does not exist. Falling back to %s "
            "(will auto-clone if missing).",
            _CHECKPOINTS_ENV_VAR,
            path,
            DEFAULT_CHECKPOINTS_DIR,
        )
    return DEFAULT_CHECKPOINTS_DIR


def get_checkpoints_version_dir(version: Optional[str] = None) -> Path:
    """Return ``<root>/<version>/`` containing ``gen/`` and ``dis/`` subdirs.

    Args:
        version: Release tag to select (e.g. ``release``). Defaults to
            :data:`DEFAULT_CHECKPOINT_VERSION`.
    """
    return get_checkpoints_root() / (version or DEFAULT_CHECKPOINT_VERSION)


def ensure_checkpoints(force: bool = False) -> Path | None:
    """Make sure the graspgenx_checkpoints repo is available locally.

    Behavior:
        * If ``$GRASPGENX_CHECKPOINT_DIR`` is set and the path exists, return
          that path with no clone.
        * Otherwise, ensure ``<repo_root>/ext/graspgenx_checkpoints`` exists,
          ``git clone``ing the upstream repo if not.
    """
    global _setup_done_checkpoints

    override = os.environ.get(_CHECKPOINTS_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return path
        logger.warning(
            "%s points to %s, which does not exist. Will auto-clone into %s.",
            _CHECKPOINTS_ENV_VAR,
            path,
            DEFAULT_CHECKPOINTS_DIR,
        )

    target = DEFAULT_CHECKPOINTS_DIR

    with _setup_lock:
        if _setup_done_checkpoints and not force:
            return target if _is_valid_clone(target) else None

        if _is_valid_clone(target) and not force:
            _setup_done_checkpoints = True
            return target

        if force and target.exists():
            logger.info("Removing existing %s for re-clone.", target)
            shutil.rmtree(target, ignore_errors=True)

        if not _git_clone_to(
            CHECKPOINTS_URL,
            target,
            dep_label="graspgenx_checkpoints",
            env_var=_CHECKPOINTS_ENV_VAR,
        ):
            return None

        _setup_done_checkpoints = True
        return target
