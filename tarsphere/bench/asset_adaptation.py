"""Runtime-neutral contract for one benchmark-case asset adaptation."""

from __future__ import annotations

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


Point3 = Tuple[float, float, float]


class StrictModel(BaseModel):
    """Immutable current-release episode schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EpisodeAssetInstance(StrictModel):
    """One exact runtime USD instance introduced by an episode."""

    object_id: str = Field(
        min_length=1,
        description="Stable object id consumed by task, trigger, and evaluator.",
    )
    role: Literal["target", "clarification"] = Field(
        description="Physical or referential role of the introduced instance.",
    )
    asset_key: str = Field(
        min_length=1,
        description="Exact category/asset_id key in the measured spawn pool.",
    )
    category: str = Field(
        min_length=1,
        description="Grounded semantic category from the spawn pool.",
    )
    asset_path: str = Field(
        min_length=1,
        description="Repository asset-relative USD payload path.",
    )
    asset_meters_per_unit: float = Field(
        gt=0.0,
        description="Authored metres represented by one asset-local unit.",
    )
    runtime_prim: str = Field(
        pattern=r"^/World/TARSphereEpisodeAssets/.+$",
        description="Unique USD prim introduced for this episode instance.",
    )
    position_xyz_m: Point3 = Field(
        description="Requested world-space instance centre in metres.",
    )
    orientation_xyzw: Tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 1.0),
        description="Requested world orientation as an XYZW quaternion.",
    )
    size_xyz_m: Point3 = Field(
        description="Measured world-space asset extents in metres.",
    )
    support_object_id: str = Field(
        min_length=1,
        description="Routed scene support receiving this instance.",
    )
    caption: str = Field(
        min_length=1,
        description="Measurement-backed language evidence for this instance.",
    )


class EpisodeAssetAdaptation(StrictModel):
    """Complete reversible scene edit referenced by one or more episodes."""

    adaptation_id: str = Field(
        min_length=1,
        description="Stable identity of this reusable physical scene edit.",
    )
    scene_id: str = Field(
        min_length=1,
        description="Base grounded scene edited at episode load time.",
    )
    routed_anchor_object_id: str = Field(
        min_length=1,
        description="Native routed object whose approach geometry is reused.",
    )
    hidden_runtime_prims: Tuple[str, ...] = Field(
        description="Native prims deactivated while adapted instances are live.",
    )
    instances: Tuple[EpisodeAssetInstance, ...] = Field(
        min_length=1,
        description=(
            "Exact assets, poses, identities, and support bindings. Base tasks "
            "may introduce only the unique target; grounding slices may add "
            "same-category clarification instances."
        ),
    )
    settle_physics_steps: int = Field(
        default=120,
        ge=1,
        description="Zero-command physics steps required before live checks.",
    )
    spawn_mass_kg: float = Field(
        default=0.1,
        gt=0.0,
        description=(
            "Temporary physical mass used while the introduced assets settle."
        ),
    )
    live_requirements: Tuple[
        Literal[
            "asset_payload_load",
            "spawn_collision_free",
            "spawn_and_settle",
            "grasp_physical_certification",
        ],
        ...,
    ] = Field(
        default=(
            "asset_payload_load",
            "spawn_collision_free",
            "spawn_and_settle",
            "grasp_physical_certification",
        ),
        description=("Live gates still required beyond deterministic asset grounding."),
    )

    @model_validator(mode="after")
    def validate_instances(self) -> "EpisodeAssetAdaptation":
        targets = [item for item in self.instances if item.role == "target"]
        if len(targets) != 1:
            raise ValueError("asset adaptation requires exactly one target")
        ids = tuple(item.object_id for item in self.instances)
        prims = tuple(item.runtime_prim for item in self.instances)
        if len(ids) != len(set(ids)) or len(prims) != len(set(prims)):
            raise ValueError("adapted object ids and runtime prims must be unique")
        if self.clarification and len({item.category for item in self.instances}) != 1:
            raise ValueError(
                "current clarification contract requires same-category assets"
            )
        roots = {item.runtime_prim.rsplit("/", 1)[0] for item in self.instances}
        if len(roots) != 1:
            raise ValueError("all adapted instances must share one family runtime root")
        return self

    @property
    def runtime_root(self) -> str:
        return self.instances[0].runtime_prim.rsplit("/", 1)[0]

    @property
    def target(self) -> EpisodeAssetInstance:
        return next(item for item in self.instances if item.role == "target")

    @property
    def clarification(self) -> Tuple[EpisodeAssetInstance, ...]:
        return tuple(item for item in self.instances if item.role == "clarification")


__all__ = ["EpisodeAssetAdaptation", "EpisodeAssetInstance"]
