"""Typed production plan for the reviewed candidate corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.contracts.task_profile import CapabilitySlice, TaskMainline


CANDIDATE_MANIFEST_VERSION = "tarsphere.candidate_manifest.v1"
EPISODE_CONTRACT = "tarsphere.bench.schema.TARSphereEpisode"


class CandidateManifestEntry(BaseModel):
    """One planned scene/cell realization and its authoritative YAML path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(
        min_length=1,
        description="Episode identity that the referenced YAML must own.",
    )
    path: str = Field(
        min_length=1,
        description="Repository-relative path to the candidate Episode YAML.",
    )
    scene_id: str = Field(
        min_length=1,
        description="Pinned scene assigned to this production work item.",
    )
    mainline: TaskMainline = Field(
        description="Physical task mainline assigned by the corpus plan.",
    )
    capability_slice: CapabilitySlice = Field(
        description="Primary capability slice assigned by the corpus plan.",
    )


class CandidateManifest(BaseModel):
    """Exact candidate set expected from one production authoring root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tarsphere.candidate_manifest.v1"] = Field(
        description="Current candidate-manifest contract version.",
    )
    episode_contract: Literal["tarsphere.bench.schema.TARSphereEpisode"] = Field(
        description="Parser authority required for every referenced YAML.",
    )
    episodes: Tuple[CandidateManifestEntry, ...] = Field(
        min_length=1,
        description="Complete planned candidate set; undeclared YAML is rejected.",
    )

    @model_validator(mode="after")
    def validate_unique_work_items(self) -> "CandidateManifest":
        identities = tuple(entry.episode_id for entry in self.episodes)
        paths = tuple(entry.path for entry in self.episodes)
        cells = tuple(
            (entry.scene_id, entry.mainline, entry.capability_slice)
            for entry in self.episodes
        )
        for field, values in (
            ("episode_id", identities),
            ("path", paths),
            ("scene/mainline/capability_slice", cells),
        ):
            if len(values) != len(set(values)):
                raise ValueError(
                    f"candidate manifest contains duplicate {field} work items"
                )
        return self


def load_candidate_manifest(path: Path) -> CandidateManifest:
    """Load the one manifest without accepting alternate or inferred shapes."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"candidate manifest does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read candidate manifest {path}: {exc}") from exc
    return CandidateManifest.model_validate(payload)


__all__ = [
    "CANDIDATE_MANIFEST_VERSION",
    "EPISODE_CONTRACT",
    "CandidateManifest",
    "CandidateManifestEntry",
    "load_candidate_manifest",
]
