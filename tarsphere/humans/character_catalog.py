"""Single-source character catalog with a strict public/private boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "assets" / "characters" / "characters.json"
PUBLIC_PROFILE_FIELDS = ("name", "caption", "job", "reference_images")
PRIVATE_EVAL_FIELDS = ("appearance_attributes_gt",)


@dataclass(frozen=True)
class CharacterCatalogCoverage:
    """Authorable identities and explicitly excluded local asset names."""

    primary_usable: tuple[str, ...]
    original_only_no_primary_twin: tuple[str, ...]
    excluded: dict[str, str]

    @property
    def authorable_assets(self) -> tuple[str, ...]:
        return self.primary_usable + self.original_only_no_primary_twin


def _load_catalog_payload() -> dict[str, Any]:
    try:
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load character catalog: {CATALOG_PATH}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Character catalog root must be a JSON object.")
    return payload


def load_character_catalog() -> dict[str, dict[str, Any]]:
    """Load and validate every usable character record."""
    payload = _load_catalog_payload()
    records: dict[str, dict[str, Any]] = {}
    for asset, raw in payload.items():
        if str(asset).startswith("_"):
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"Character catalog record must be an object: {asset!r}")
        record = dict(raw)
        for field in ("name", "caption", "job", "isaac_category", "usd_file"):
            if not str(record.get(field) or "").strip():
                raise ValueError(f"Character {asset!r} missing required {field!r}.")
        usd_file = Path(str(record["usd_file"]))
        if (
            usd_file.name != str(record["usd_file"])
            or usd_file.suffix.casefold() != ".usd"
        ):
            raise ValueError(
                f"Character {asset!r} usd_file must be one relative USD filename."
            )
        for field in ("reference_images", "appearance_attributes_gt"):
            value = record.get(field)
            if not isinstance(value, list) or not value:
                raise ValueError(f"Character {asset!r} requires non-empty {field!r}.")
            if not all(isinstance(item, str) and item.strip() for item in value):
                raise ValueError(f"Character {asset!r} {field!r} must contain strings.")
        records[str(asset)] = record

    names: set[str] = set()
    for asset, record in records.items():
        name_key = str(record["name"]).casefold()
        if name_key in names:
            raise ValueError(f"Character names must be unique: {record['name']!r}")
        names.add(name_key)
    return records


def character_catalog_coverage() -> CharacterCatalogCoverage:
    """Load the current authoring eligibility policy from ``characters.json``."""
    payload = _load_catalog_payload()
    raw = payload.get("_coverage")
    if not isinstance(raw, dict):
        raise ValueError("Character catalog requires a _coverage object.")

    def _asset_list(field: str) -> tuple[str, ...]:
        values = raw.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Character catalog _coverage.{field} must be non-empty.")
        normalized = tuple(str(value).strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError(f"Character catalog _coverage.{field} has empty values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Character catalog _coverage.{field} has duplicates.")
        if normalized != tuple(sorted(normalized)):
            raise ValueError(f"Character catalog _coverage.{field} must be sorted.")
        return normalized

    primary = _asset_list("primary_usable")
    original_only = _asset_list("original_only_no_primary_twin")
    if set(primary) & set(original_only):
        raise ValueError("Character catalog authorable coverage sets must be disjoint.")

    excluded_raw = raw.get("excluded")
    if not isinstance(excluded_raw, dict):
        raise ValueError("Character catalog _coverage.excluded must be an object.")
    excluded = {
        str(asset).strip(): str(reason).strip()
        for asset, reason in excluded_raw.items()
    }
    if any(not asset or not reason for asset, reason in excluded.items()):
        raise ValueError(
            "Character catalog exclusions require asset names and reasons."
        )

    records = load_character_catalog()
    canonical = set(records)
    declared = set(primary) | set(original_only)
    if declared != canonical:
        raise ValueError(
            "Character catalog coverage must declare every canonical identity "
            "exactly once."
        )
    return CharacterCatalogCoverage(
        primary_usable=primary,
        original_only_no_primary_twin=original_only,
        excluded=excluded,
    )


def character_record(asset: str) -> dict[str, Any]:
    """Return the full evaluator-side record for one episode asset."""
    records = load_character_catalog()
    try:
        return dict(records[str(asset)])
    except KeyError as exc:
        raise KeyError(f"Unknown character asset: {asset!r}") from exc


def public_character_profiles(
    names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return only agent-visible fields; evaluator goldens never cross here."""
    requested = {
        str(name).strip().casefold() for name in (names or []) if str(name).strip()
    }
    profiles: list[dict[str, Any]] = []
    matched: set[str] = set()
    for asset, record in load_character_catalog().items():
        name = str(record["name"])
        if requested and name.casefold() not in requested:
            continue
        matched.add(name.casefold())
        profile = {"asset": asset}
        for field in PUBLIC_PROFILE_FIELDS:
            value = record[field]
            profile[field] = list(value) if isinstance(value, list) else value
        profiles.append(profile)
    missing = requested - matched
    if missing:
        raise KeyError(f"Unknown public character name(s): {sorted(missing)}")
    return profiles


def public_character_profiles_for_assets(
    assets: list[str],
) -> list[dict[str, Any]]:
    """Return public profiles for exactly the characters used by an episode."""
    records = load_character_catalog()
    profiles: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_asset in assets:
        asset = str(raw_asset).strip()
        if not asset:
            raise ValueError("Episode character asset names must be non-empty.")
        try:
            record = records[asset]
        except KeyError as exc:
            raise KeyError(f"Unknown character asset: {asset!r}") from exc
        name_key = str(record["name"]).casefold()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        profile = {"asset": asset}
        for field in PUBLIC_PROFILE_FIELDS:
            value = record[field]
            profile[field] = list(value) if isinstance(value, list) else value
        profiles.append(profile)
    return profiles


def resolve_reference_image(path: str) -> Path:
    """Resolve one catalog image and fail loud if the official asset is absent."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"Character reference image not found: {candidate}")
    return candidate


def resolve_character_usd(
    asset: str,
    *,
    character_assets_root: Path | str | None = None,
) -> Path:
    """Resolve the one catalog-declared USD for an authorable Human asset."""
    if character_assets_root is None:
        from tarsphere.assets import tarsphere_assets_root

        root = tarsphere_assets_root() / "characters" / "IsaacPeople"
    else:
        root = Path(character_assets_root).expanduser().resolve()
    record = character_record(asset)
    candidate = root / str(asset) / str(record["usd_file"])
    if not candidate.is_file():
        raise FileNotFoundError(
            "Character USD declared by "
            f"assets/characters/characters.json is missing: "
            f"{candidate}"
        )
    return candidate


__all__ = [
    "CATALOG_PATH",
    "CharacterCatalogCoverage",
    "PRIVATE_EVAL_FIELDS",
    "PUBLIC_PROFILE_FIELDS",
    "character_catalog_coverage",
    "character_record",
    "load_character_catalog",
    "public_character_profiles",
    "public_character_profiles_for_assets",
    "resolve_character_usd",
    "resolve_reference_image",
]
