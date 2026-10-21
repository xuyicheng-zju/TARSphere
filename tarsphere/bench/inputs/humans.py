"""Verified Human identities, assets, and runtime capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict

from tarsphere.humans.character_catalog import (
    character_catalog_coverage,
    load_character_catalog,
    resolve_character_usd,
    resolve_reference_image,
)
from tarsphere.humans.runtime_capabilities import (
    HUMAN_RUNTIME_CAPABILITIES,
    HumanRuntimeCommand,
    resolve_isaac_people_runtime_assets,
)
from tarsphere.assets import tarsphere_assets_root


class HumanAssetCatalogError(RuntimeError):
    """Raised when local Human assets contradict ``characters.json``."""


class HumanPublicProfile(BaseModel):
    """Agent-visible identity fields for one episode-scoped profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    job: str
    reference_images: Tuple[str, ...]


class HumanAppearanceGroundTruth(BaseModel):
    """Evaluator-private appearance facts; never exposed by the profile tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    caption: str
    attributes: Tuple[str, ...]


class HumanAssetEntry(BaseModel):
    """One canonical, physically present Human identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    isaac_category: str
    asset_usd: str
    public_profile: HumanPublicProfile
    evaluator_private: HumanAppearanceGroundTruth


class HumanAssetExclusion(BaseModel):
    """One local support directory that is not an episode Human."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    reason: str


class HumanAssetCatalog(BaseModel):
    """Current-only cast inventory presented to the episode authoring plane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authorable_assets: Tuple[HumanAssetEntry, ...]
    excluded_assets: Tuple[HumanAssetExclusion, ...]

    def require_asset(self, asset_id: str) -> HumanAssetEntry:
        for entry in self.authorable_assets:
            if entry.asset_id == asset_id:
                return entry
        raise KeyError(
            f"Human asset is not authorable in the current release: {asset_id!r}"
        )


class HumanAnimationClip(BaseModel):
    """One required physical SkelAnimation asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    asset_path: str


class HumanRuntimeCatalog(BaseModel):
    """Commands and verified animation assets available to episode designers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commands: Tuple[HumanRuntimeCommand, ...]
    animation_clips: Tuple[HumanAnimationClip, ...]

    def require_command(self, name: str) -> HumanRuntimeCommand:
        for command in self.commands:
            if command.name == name:
                return command
        raise KeyError(f"Human runtime command is not authorable: {name!r}")


def build_human_asset_catalog(
    *,
    assets_root: Optional[Path | str] = None,
) -> HumanAssetCatalog:
    """Build the 11-identity Human catalog from declared, physical assets."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root
        else tarsphere_assets_root()
    )
    character_root = root / "characters" / "IsaacPeople"
    if not character_root.is_dir():
        raise FileNotFoundError(
            f"Isaac People character root is missing: {character_root}"
        )

    records = load_character_catalog()
    coverage = character_catalog_coverage()
    authorable: list[HumanAssetEntry] = []
    for asset_id in coverage.authorable_assets:
        record = records[asset_id]
        asset_usd = resolve_character_usd(
            asset_id,
            character_assets_root=character_root,
        )
        for reference in record["reference_images"]:
            resolve_reference_image(str(reference))
        authorable.append(
            HumanAssetEntry(
                asset_id=asset_id,
                isaac_category=str(record.get("isaac_category") or ""),
                asset_usd=_relative_from_mount(
                    asset_usd,
                    root=root,
                    mount=character_root,
                ),
                public_profile=HumanPublicProfile(
                    name=str(record["name"]),
                    job=str(record["job"]),
                    reference_images=tuple(
                        str(reference) for reference in record["reference_images"]
                    ),
                ),
                evaluator_private=HumanAppearanceGroundTruth(
                    caption=str(record["caption"]),
                    attributes=tuple(
                        str(attribute)
                        for attribute in record["appearance_attributes_gt"]
                    ),
                ),
            )
        )

    authorable_ids = set(coverage.authorable_assets)
    excluded: list[HumanAssetExclusion] = []
    for directory in sorted(
        path
        for path in character_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ):
        if directory.name in authorable_ids:
            continue
        reason = coverage.excluded.get(directory.name)
        if not reason:
            raise HumanAssetCatalogError(
                f"Unregistered Human asset directory cannot be silently "
                f"excluded: {directory}"
            )
        excluded.append(
            HumanAssetExclusion(
                asset_id=directory.name,
                reason=reason,
            )
        )

    return HumanAssetCatalog(
        authorable_assets=tuple(authorable),
        excluded_assets=tuple(excluded),
    )


def build_human_runtime_catalog(
    *,
    assets_root: Optional[Path | str] = None,
) -> HumanRuntimeCatalog:
    """Verify the command contract against the installed Isaac People clips."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root
        else tarsphere_assets_root()
    )
    runtime_assets = resolve_isaac_people_runtime_assets(
        root / "characters" / "IsaacPeople"
    )
    animation_root = runtime_assets.animation_directory

    clips: list[HumanAnimationClip] = []
    for name in HUMAN_RUNTIME_CAPABILITIES.required_animation_clips:
        path = animation_root / f"{name}.skelanim.usd"
        if not path.is_file():
            raise FileNotFoundError(f"Human runtime animation clip is missing: {path}")
        clips.append(
            HumanAnimationClip(
                name=name,
                asset_path=_relative_from_mount(
                    path,
                    root=root,
                    mount=root / "characters" / "IsaacPeople",
                ),
            )
        )

    return HumanRuntimeCatalog(
        commands=HUMAN_RUNTIME_CAPABILITIES.commands,
        animation_clips=tuple(clips),
    )


def _relative_from_mount(
    path: Path,
    *,
    root: Path,
    mount: Path,
) -> str:
    """Return a portable asset path through a declared mounted subtree.

    ``IsaacPeople`` may be a symlink to the machine-local staged asset tree.
    Security is therefore checked against the resolved mount target, while the
    serialized path keeps the repository-relative mount name.
    """
    try:
        relative_inside_mount = path.resolve().relative_to(mount.resolve())
        logical_mount = mount.relative_to(root)
    except ValueError as exc:
        raise HumanAssetCatalogError(
            f"Human asset escaped configured mount {mount}: {path}"
        ) from exc
    return (logical_mount / relative_inside_mount).as_posix()


__all__ = [
    "HumanAnimationClip",
    "HumanAppearanceGroundTruth",
    "HumanAssetCatalog",
    "HumanAssetCatalogError",
    "HumanAssetEntry",
    "HumanAssetExclusion",
    "HumanPublicProfile",
    "HumanRuntimeCatalog",
    "build_human_asset_catalog",
    "build_human_runtime_catalog",
]
