"""Canonical access to physically certified G1 manipulation assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CERTIFIED_ASSET_POOL = (
    REPOSITORY_ROOT
    / "episodes"
    / "production"
    / "assets"
    / "g1_certified_pickables"
    / "pool.json"
)
CERTIFIED_ASSET_POOL_VERSION = "tarsphere.g1_certified_asset_pool.v1"


def load_certified_asset_pool(
    path: Path = DEFAULT_CERTIFIED_ASSET_POOL,
) -> dict[str, Any]:
    """Load and validate the sole production source of certified assets."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: certified asset pool must be a JSON object")
    if payload.get("schema_version") != CERTIFIED_ASSET_POOL_VERSION:
        raise ValueError(f"{path}: unsupported certified asset pool version")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise TypeError(f"{path}: assets must be a JSON list")
    keys: list[str] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise TypeError(f"{path}: assets[{index}] must be a JSON object")
        key = asset.get("asset_key")
        usd_path = asset.get("usd_path")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path}: assets[{index}].asset_key is invalid")
        if not isinstance(usd_path, str) or not usd_path:
            raise ValueError(f"{path}: assets[{index}].usd_path is invalid")
        if asset.get("certification_status") != "certified":
            raise ValueError(f"{path}: {key} is not physically certified")
        certificates = asset.get("certificates")
        if not isinstance(certificates, list) or not certificates:
            raise ValueError(f"{path}: {key} has no physical certification evidence")
        if asset.get("certificate_count") != len(certificates):
            raise ValueError(
                f"{path}: {key} certificate_count does not match certificates"
            )
        valid_certificates = [
            certificate
            for certificate in certificates
            if isinstance(certificate, dict)
            and certificate.get("hand") == "right"
            and certificate.get("transport_ready") is True
            and certificate.get("retention") == "finite_coulomb_friction"
            and certificate.get("constraint_joint_present") is False
            and bool(str(certificate.get("result_path") or "").strip())
        ]
        if not valid_certificates:
            raise ValueError(
                f"{path}: {key} lacks a valid right-hand transport certificate"
            )
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate asset_key values")
    if payload.get("asset_count") != len(assets):
        raise ValueError(f"{path}: asset_count does not match assets")
    return payload


def select_certified_assets(
    pool: dict[str, Any],
    *,
    source_collection: str | None = None,
    categories: Iterable[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return a deterministic subset of physically certified assets."""

    category_set = None if categories is None else frozenset(categories)
    selected = [
        asset
        for asset in pool["assets"]
        if (
            source_collection is None or asset["source_collection"] == source_collection
        )
        and (category_set is None or asset["category"] in category_set)
    ]
    return tuple(sorted(selected, key=lambda asset: asset["asset_key"]))


def certified_asset(
    pool: dict[str, Any],
    asset_key: str,
) -> dict[str, Any]:
    """Resolve one certified asset or fail instead of falling back to raw USD."""

    matches = [asset for asset in pool["assets"] if asset["asset_key"] == asset_key]
    if len(matches) != 1:
        raise KeyError(
            f"certified asset is absent from the canonical pool: {asset_key}"
        )
    return matches[0]


__all__ = [
    "CERTIFIED_ASSET_POOL_VERSION",
    "DEFAULT_CERTIFIED_ASSET_POOL",
    "certified_asset",
    "load_certified_asset_pool",
    "select_certified_assets",
]
