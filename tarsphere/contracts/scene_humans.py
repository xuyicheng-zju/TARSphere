"""Audited scene-native Human assets.

GRScenes normally stores people below ``Furnitures/person``.  A small number
of visibly human, task-useful static assets are misclassified by the source
taxonomy.  Keep those exceptions exact and reviewable instead of accepting
arbitrary ``other`` geometry as a Human.
"""

from __future__ import annotations


REVIEWED_MISCATEGORIZED_SCENE_HUMAN_PRIMS = frozenset(
    {
        (
            "/Root/Meshes/Furnitures/other/"
            "model_271ad4de75edcb16a93ecec9a53709c7_0/Instance"
        ),
    }
)


def is_audited_scene_human_prim(path: str, *, category: str | None = None) -> bool:
    """Return whether an exact pinned prim may back a scene-bound Human."""
    normalized = str(path).strip()
    if normalized.startswith("/Root/Meshes/Furnitures/person/"):
        return category in (None, "person")
    return normalized in REVIEWED_MISCATEGORIZED_SCENE_HUMAN_PRIMS
