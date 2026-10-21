"""Compile the minimal grounded object set required by one runtime episode."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from tarsphere.bench.compilation.schema import EpisodeObject
from tarsphere.bench.grounding import GroundedScene


def compile_episode_objects(
    scene: GroundedScene,
    object_ids: Iterable[str],
    *,
    names: Mapping[str, str] | None = None,
) -> tuple[EpisodeObject, ...]:
    """Retain exact grounded objects in the requested order."""
    objects = {item.object_id: item for item in scene.objects}
    compiled = []
    seen = set()
    for object_id in object_ids:
        if object_id in seen:
            continue
        seen.add(object_id)
        try:
            obj = objects[object_id]
        except KeyError as exc:
            raise ValueError(
                f"Runtime episode references an unknown object: {object_id!r}"
            ) from exc
        if not obj.runtime_prim:
            raise ValueError(f"Runtime object has no verified prim: {object_id!r}")
        name = (names or {}).get(object_id)
        compiled.append(
            EpisodeObject(
                object_id=object_id,
                name=name,
                category=obj.category,
                room_id=obj.room_id,
                position_xyz=obj.position_xyz,
                bounds_min_xyz=obj.bounds_min_xyz,
                bounds_max_xyz=obj.bounds_max_xyz,
                runtime_prim=obj.runtime_prim,
                interaction=obj.interaction,
            )
        )
    if not compiled:
        raise ValueError("Runtime episode requires at least one grounded object")
    return tuple(compiled)


__all__ = ["compile_episode_objects"]
