"""Verified scene and Human inputs for benchmark authoring tools."""

from tarsphere.bench.inputs.humans import (
    HumanAnimationClip,
    HumanAppearanceGroundTruth,
    HumanAssetCatalog,
    HumanAssetCatalogError,
    HumanAssetEntry,
    HumanAssetExclusion,
    HumanPublicProfile,
    HumanRuntimeCatalog,
    build_human_asset_catalog,
    build_human_runtime_catalog,
)
from tarsphere.bench.inputs.scenes import GRSceneMetadata
from tarsphere.bench.inputs.sources import (
    AssetSourceRegistry,
    GRScenesCatalog,
    SceneCatalogEntry,
    build_grscenes_catalog,
    load_asset_source_registry,
)

__all__ = [
    "AssetSourceRegistry",
    "GRSceneMetadata",
    "GRScenesCatalog",
    "HumanAnimationClip",
    "HumanAppearanceGroundTruth",
    "HumanAssetCatalog",
    "HumanAssetCatalogError",
    "HumanAssetEntry",
    "HumanAssetExclusion",
    "HumanPublicProfile",
    "HumanRuntimeCatalog",
    "SceneCatalogEntry",
    "build_grscenes_catalog",
    "build_human_asset_catalog",
    "build_human_runtime_catalog",
    "load_asset_source_registry",
]
