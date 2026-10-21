"""Portable path resolution for the end2end pipeline.

No absolute paths live in committed configs — the repo can be cloned
anywhere. Robot YAMLs reference external assets with ``${...}`` tokens that
:func:`expand` resolves at load time:

  ``${CUROBO_ASSETS}``  cuRobo's shipped ``content/assets`` dir, discovered
                        from the installed ``nvidia-curobo`` package wherever
                        it lives (sibling checkout or site-packages).
  ``${GRIPPERS}``       the ``gripper_descriptions`` ``x_grippers`` dir,
                        resolved via GraspGenX (honors
                        ``$GRASPGENX_GRIPPER_CFG_DIR`` or the auto-cloned
                        ``ext/gripper_descriptions``).
  ``${E2E}``            this ``end2end/`` directory.
  ``${REPO}``           the GraspGenX repo root.

``load_yaml`` runs every loaded config through :func:`expand`, so configs
without ``${`` tokens (e.g. env YAMLs with repo-relative mesh paths) pass
through untouched and the heavy cuRobo/GraspGenX imports only fire when a
config actually references those roots.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

E2E_DIR = Path(__file__).resolve().parent
REPO_ROOT = E2E_DIR.parent


@lru_cache(maxsize=None)
def curobo_assets_dir() -> Path:
    """cuRobo's ``content/assets`` dir (robot URDFs + meshes).

    cuRobo's *installed wheel* ships only a partial asset set (no meshes, no
    ``ur_description``), so prefer a full source checkout cloned into
    ``ext/curobo`` by ``end2end/setup_end2end_deps.py`` (or pointed at via
    ``$GRASPGENX_CUROBO_DIR``). Fall back to the installed package only if no
    source checkout is available.
    """
    candidates = []
    override = os.environ.get("GRASPGENX_CUROBO_DIR")
    if override:
        candidates.append(Path(override) / "curobo/content/assets")
    candidates.append(REPO_ROOT / "ext/curobo/curobo/content/assets")
    for c in candidates:
        if (c / "robot/ur_description/ur10e.urdf").is_file():
            return c
    from curobo.content import get_assets_path

    return Path(get_assets_path())


@lru_cache(maxsize=None)
def grippers_dir() -> Path:
    """The ``gripper_descriptions`` ``x_grippers`` dir (GraspGenX-resolved)."""
    from graspgenx import get_gripper_descriptions_assets

    return Path(get_gripper_descriptions_assets())


# Token name -> zero-arg resolver. Lazily evaluated so a config that never
# uses ${CUROBO_ASSETS}/${GRIPPERS} never imports cuRobo/GraspGenX.
_RESOLVERS = {
    "${E2E}": lambda: str(E2E_DIR),
    "${REPO}": lambda: str(REPO_ROOT),
    "${CUROBO_ASSETS}": lambda: str(curobo_assets_dir()),
    "${GRIPPERS}": lambda: str(grippers_dir()),
}


def expand(value: Any) -> Any:
    """Recursively expand ``${...}`` path tokens in str / list / dict values."""
    if isinstance(value, str):
        if "${" not in value:
            return value
        out = value
        for token, resolve in _RESOLVERS.items():
            if token in out:
                out = out.replace(token, resolve())
        return out
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value
