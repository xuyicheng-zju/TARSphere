"""Single current path authority for TARSphere assets."""

from __future__ import annotations

import os
from pathlib import Path


TARSPHERE_ASSETS_ENV = "TARSPHERE_ASSETS_PATH"

TARSPHERE_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_ROOT = TARSPHERE_REPOSITORY_ROOT / "assets"


def tarsphere_assets_root(*, require: bool = True) -> Path:
    """Return the one TARSphere asset root."""
    configured = os.environ.get(TARSPHERE_ASSETS_ENV)
    root = (
        Path(configured).expanduser() if configured else DEFAULT_ASSETS_ROOT
    ).resolve()
    if require and not root.is_dir():
        raise FileNotFoundError(
            f"TARSphere assets not found at {root}. Create the asset tree under "
            f"{DEFAULT_ASSETS_ROOT} or set {TARSPHERE_ASSETS_ENV}."
        )
    return root


def configure_simulator_assets() -> Path:
    """Point the built-in TARSphere simulator at TARSPHERE's flat asset root."""
    root = tarsphere_assets_root(require=True)
    os.environ[TARSPHERE_ASSETS_ENV] = str(root)

    from tarsphere.simulator.macros import gm

    current = gm.get("ASSET_PATH")
    if current and Path(str(current)).expanduser().resolve() == root:
        return root

    try:
        gm.ASSET_PATH = str(root)
    except AttributeError as exc:
        raise RuntimeError(
            "tarsphere.simulator.macros.gm.ASSET_PATH was already read before TARSphere "
            f"could point it at {root}. Import tarsphere.assets before "
            "importing tarsphere.simulator.extensions configs."
        ) from exc
    return root


def tarsphere_asset_path(*parts: str) -> Path:
    """Return a path inside the configured TARSphere asset root."""
    return tarsphere_assets_root(require=True).joinpath(*parts)


def resolve_tarsphere_asset(asset_path: str | os.PathLike[str]) -> Path:
    """Resolve an asset path against the configured TARSphere asset root."""
    root = tarsphere_assets_root(require=True)
    path = Path(str(asset_path).strip()).expanduser()
    return path if path.is_absolute() else root / path


__all__ = [
    "DEFAULT_ASSETS_ROOT",
    "TARSPHERE_ASSETS_ENV",
    "TARSPHERE_REPOSITORY_ROOT",
    "configure_simulator_assets",
    "tarsphere_asset_path",
    "tarsphere_assets_root",
    "resolve_tarsphere_asset",
]
