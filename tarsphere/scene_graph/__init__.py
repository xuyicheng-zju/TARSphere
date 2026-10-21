"""Scene-level GRScene graph assets."""

from .builder import build_scenegraph, default_scenegraph_output_path, write_scenegraph
from .room_annotations import (
    RoomOwnershipRegion,
    build_room_annotations,
    generate_pinned_room_annotation_assets,
    generate_room_annotation_asset,
    load_room_ownership_regions,
    render_room_annotation_map,
    resolve_room_owner_id,
)

__all__ = [
    "build_scenegraph",
    "build_room_annotations",
    "default_scenegraph_output_path",
    "generate_pinned_room_annotation_assets",
    "generate_room_annotation_asset",
    "load_room_ownership_regions",
    "render_room_annotation_map",
    "resolve_room_owner_id",
    "RoomOwnershipRegion",
    "write_scenegraph",
]
