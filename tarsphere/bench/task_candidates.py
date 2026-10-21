"""Deterministic inputs for physical task candidate search.

These models are tool inputs, not an alternative episode schema.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PhysicalTaskProposal(StrictModel):
    """One grounded target and destination awaiting certification."""

    target_object_id: str = Field(min_length=1)
    destination_object_id: str = Field(min_length=1)
    goal_relation: Literal["on"]

    @model_validator(mode="after")
    def validate_distinct_roles(self) -> "PhysicalTaskProposal":
        if self.target_object_id == self.destination_object_id:
            raise ValueError("target and destination must be distinct")
        return self


class PhysicalTaskShortlist(StrictModel):
    """Ordered grounded proposals for one scene."""

    scene_id: str = Field(min_length=1)
    proposals: Tuple[PhysicalTaskProposal, ...] = Field(
        min_length=1,
        max_length=5,
    )

    @model_validator(mode="after")
    def validate_unique_proposals(self) -> "PhysicalTaskShortlist":
        identities = tuple(
            (
                proposal.target_object_id,
                proposal.destination_object_id,
                proposal.goal_relation,
            )
            for proposal in self.proposals
        )
        duplicates = [
            identity for identity, count in Counter(identities).items() if count > 1
        ]
        if duplicates:
            raise ValueError(f"duplicate physical-task proposals: {duplicates}")
        return self


__all__ = ["PhysicalTaskProposal", "PhysicalTaskShortlist"]
