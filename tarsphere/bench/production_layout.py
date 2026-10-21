"""Conservative production placement screens; never physical certification.

This owns production-only authoring margins. Runtime reach limits and task
scoring remain unchanged. Masks are projected from the evaluation scene's
active meshes, including instance proxies and tall furniture, not room labels.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from tarsphere.contracts.human_interaction import HUMAN_COMMUNICATION_RADIUS_M
from tarsphere.contracts.scene_humans import is_audited_scene_human_prim
from tarsphere.bench.schema import INITIAL_REQUEST_FACING_TOLERANCE_RAD

# Candidate-search envelope, not physical certification or universal G1 limits.
# The pool's full physical certificates include 0.550 m (e.g. the onion's
# episode_0.json). High counters can contact neutral hands at closer stances.
# Every authored placement still needs scene-specific SAM3 and physical grasp;
# a certificate at a lower worktop does not certify this envelope at all heights.
FETCH_STANCE_DISTANCE_BAND_M = (0.40, 0.55)
# Prefer ordinary worktops; exclude crouch-heavy chair/coffee-table placements.
FETCH_SUPPORT_HEIGHT_BAND_M = (0.65, 0.95)
FETCH_TARGET_MAX_TOP_M = 1.30
# Minimum meaningful route extent for this production portfolio, not hidden
# waypoints or runtime stopping tolerances.
MIN_INSPECTION_OUTBOUND_LENGTH_M = 3.0
MIN_FETCH_NAVIGATE_TO_GRASP_LENGTH_M = 1.5
MIN_COMPLETION_ROUTE_LENGTH_M = 3.0
# Runtime communication remains possible out to 2.5 m, but authored onsite
# and completion stances should represent an ordinary face-to-face exchange.
# Keep that review contract separate from the permissive runtime gate.
PREFERRED_REVIEW_HUMAN_INTERACTION_DISTANCE_M = 2.0
# Scene-native people are part of the authored world, not decorative pixels.
# In commercial scenes, spawning a catalog actor at the same semantic locus
# creates an implausible crowd and wastes an already grounded interaction role.
MIN_INTERACTABLE_SCENE_PERSON_HEIGHT_M = 0.60
MIN_CATALOG_NATIVE_PERSON_SEPARATION_M = 2.0
# The runtime footprint already contributes 0.25 m of obstacle clearance.
# Keep an additional measured margin at A so the normal G1 reset/settle drift
# cannot move the first live pose outside the exact mask used by PointGoal.
MIN_ROUTE_START_EXTRA_CLEARANCE_M = 0.05
MAX_BASE_NPCS = 3
FETCH_FOOT_CLEARANCE_MARGIN_M = 0.01
FETCH_SUPPORT_EDGE_MARGIN_M = 0.02
SUPPORT_SAMPLE_CLEARANCE_M = 0.08
SUPPORT_SAMPLE_POSITION_TOLERANCE_M = 0.012
SUPPORT_SAMPLE_HEIGHT_TOLERANCE_M = 0.006
SUPPORTED_WORKTOP_CATEGORIES = frozenset(
    {"table", "desk", "counter", "cabinet", "shelf", "bookshelf", "tray", "other"}
)


class SiteInteraction(BaseModel):
    """Deterministic dialogue with the addressable Human at the B stance.

    Inspection tasks may use this as optional supporting dialogue.  Inquiry
    tasks use the delivered answer as their canonical onsite task outcome.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    human_id: str = Field(
        min_length=1,
        description="Declared NPC present and addressable at the objective stance.",
    )
    intent_id: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Episode intent identifier exposed to the public ask tool.",
    )
    description: str = Field(
        min_length=1,
        description="Public question description, not the inspection answer.",
    )
    answer: str = Field(
        min_length=1,
        description="Deterministic private NPC reply; must not disclose the scored visual observation.",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world_pixels(points, meta):
    """Native BEV projection, rejecting out-of-frame points instead of clipping."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    lo = np.asarray(meta["bev_bounds_min_xy"], dtype=float)
    span = float(meta["bev_span_m"])
    size = int(meta["resolution"])
    pixels = np.column_stack(
        ((points[:, 0] - lo[0]) / span, (lo[1] + span - points[:, 1]) / span)
    ) * (size - 1)
    if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels > size - 1):
        raise ValueError("Layout geometry projects outside the native BEV")
    return pixels


def segment_clear(mask, start, end, meta, *, terminal_clearance_m=0.0):
    """Supercover a ray, optionally excluding its scene-bound actor endpoint.

    A scene-native Human is already painted into the projected sight mask. A
    ray aimed at that Human's exact center must ignore only the terminal body
    volume, while retaining every blocker before it.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if terminal_clearance_m < 0 or not math.isfinite(terminal_clearance_m):
        raise ValueError("terminal_clearance_m must be finite and non-negative")
    pixels = world_pixels([start, end], meta)
    count = max(2, int(np.ceil(np.max(np.abs(pixels[1] - pixels[0])) * 4)) + 1)
    fractions = np.linspace(0.0, 1.0, count)
    samples = pixels[0] + fractions[:, None] * (pixels[1] - pixels[0])
    distance = float(np.linalg.norm(end - start))
    if terminal_clearance_m and distance:
        samples = samples[(1.0 - fractions) * distance > terminal_clearance_m]
    if not len(samples):
        return True
    lower = np.floor(samples).astype(int)
    upper = np.ceil(samples).astype(int)
    return not any(
        mask[y, x].any()
        for x in (lower[:, 0], upper[:, 0])
        for y in (lower[:, 1], upper[:, 1])
    )


@lru_cache(maxsize=4)
def _read_contact_profile(path, sha):
    profile = json.loads(path.read_text())
    if profile["schema_version"] != "tarsphere.g1_lower_body_profile.v1":
        raise ValueError("Unsupported G1 contact profile")
    return profile


def load_contact_profile(path: Path | str, *, source_root: Path | str | None = None):
    """Read a user-supplied authoring profile and verify its declared sources.

    Relative source paths resolve against ``source_root`` when supplied, or
    against the profile's directory. No generated profile is bundled.
    """
    from tarsphere.bench.certification.scene import _file_digest

    path = Path(path).expanduser().resolve()
    root = Path(source_root).expanduser().resolve() if source_root else path.parent
    profile = _read_contact_profile(path, digest(path))
    for source, sha in profile.get("source_hashes", {}).items():
        source_path = root / source
        if _file_digest(source_path) != sha:
            raise ValueError(f"Stale G1 lower-body profile: {source}")
    return profile


def foot_placement_clear(mask, stance, target, meta, *, profile):
    """Screen an oriented nominal low-body collision envelope, not whole-body XY.

    The profile is derived from the harness's composed robot and nominal leg
    pose, ground aligned. It does not certify the dynamic grasp trajectory.
    """
    stance = np.asarray(stance, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - stance
    distance = np.linalg.norm(delta)
    if distance < 1e-8:
        return False
    forward = delta / distance
    left = np.array([-forward[1], forward[0]])
    margin = FETCH_FOOT_CLEARANCE_MARGIN_M
    low = np.asarray(profile["bounds_min_xy_m"]) - margin
    high = np.asarray(profile["bounds_max_xy_m"]) + margin
    step = float(meta["bev_span_m"]) / (int(meta["resolution"]) - 1) / 2
    xs = np.linspace(low[0], high[0], int(np.ceil((high[0] - low[0]) / step)) + 1)
    ys = np.linspace(low[1], high[1], int(np.ceil((high[1] - low[1]) / step)) + 1)
    x, y = np.meshgrid(xs, ys)
    points = stance + x.ravel()[:, None] * forward + y.ravel()[:, None] * left
    pixels = world_pixels(points, meta)
    lower = np.floor(pixels).astype(int)
    upper = np.ceil(pixels).astype(int)
    return not any(
        mask[y, x].any()
        for x in (lower[:, 0], upper[:, 0])
        for y in (lower[:, 1], upper[:, 1])
    )


@lru_cache(maxsize=8)
def _read_geometry(path: Path, sha: str, sight_sha: str, feet_sha: str):
    geometry = json.loads(path.read_text())
    masks = {}
    for name in ("sight", "feet"):
        mask_path = path.with_suffix(f".{name}.png")
        if {"sight": sight_sha, "feet": feet_sha}[name] != geometry[
            f"{name}_mask_sha256"
        ]:
            raise ValueError(f"Stale {name} geometry mask: {mask_path}")
        masks[name] = np.asarray(Image.open(mask_path)) > 0
    if (
        geometry["support_sampling"]
        != "mesh_surface_80mm_clearance_6mm_flatness_10mm_raster"
    ):
        raise ValueError(f"Stale support sampling contract: {path}")
    return geometry, masks


def load_layout_geometry(
    path, meta, projection_path, *, root=Path(__file__).resolve().parents[2]
):
    path = Path(path)
    geometry, masks = _read_geometry(
        path,
        digest(path),
        digest(path.with_suffix(".sight.png")),
        digest(path.with_suffix(".feet.png")),
    )
    if geometry["projection_sha256"] != digest(Path(projection_path)):
        raise ValueError("Layout geometry BEV projection is stale")
    if (
        geometry.get("geometry_contract") != "active_mesh_layout_v1"
        or geometry["scene_id"] != Path(projection_path).parent.name
    ):
        raise ValueError("Layout geometry contract or scene identity is invalid")
    if geometry.get("instance_collections") != [
        "Furnitures",
        "Animation",
        "BaseAnimation",
        "Base",
    ]:
        raise ValueError(
            "Layout support inventory omits required scene instance collections"
        )
    source = root / geometry["source_usd"]
    if (
        source.resolve()
        != (Path(projection_path).parent / "start_result_navigation.usd").resolve()
    ):
        raise ValueError("Layout belongs to a different evaluation USD")
    if digest(source) != geometry["source_sha256"]:
        raise ValueError("Layout evaluation USD is stale")
    provenance_path = root / geometry["navigation_provenance"]
    if digest(provenance_path) != geometry["navigation_provenance_sha256"]:
        raise ValueError("Layout navigation provenance is stale")
    provenance = json.loads(provenance_path.read_text())
    if (
        provenance["source_sha256"] != geometry["source_sha256"]
        or provenance["scene_id"] != geometry["scene_id"]
    ):
        raise ValueError(
            "Layout and navigation evidence belong to different scene revisions"
        )
    from tarsphere.bench.certification.scene import validate_scene_occupancy_provenance

    validate_scene_occupancy_provenance(
        root / provenance["occupancy"],
        root=root,
        scene_usd=source,
        verify_scene_files=False,
    )
    if any(m.shape != (int(meta["resolution"]),) * 2 for m in masks.values()):
        raise ValueError("Layout masks must preserve native BEV resolution")
    return geometry, masks


def _support_patch_covers_footprint(center, size, samples):
    """Cover the full margin-expanded footprint by measured coplanar patches.

    The producer certifies an axis-aligned 80 mm half-width square around
    each sample. Adjacent patches may jointly support a book or medicine box;
    a large furniture AABB alone is never evidence of continuous support.
    Partition at every patch edge so coverage is exact, including interior holes.
    """
    low = center - size / 2 - FETCH_SUPPORT_EDGE_MARGIN_M
    high = center + size / 2 + FETCH_SUPPORT_EDGE_MARGIN_M
    patch_low = samples[:, :2] - SUPPORT_SAMPLE_CLEARANCE_M
    patch_high = samples[:, :2] + SUPPORT_SAMPLE_CLEARANCE_M
    relevant = np.all(patch_high >= low, axis=1) & np.all(patch_low <= high, axis=1)
    patch_low, patch_high = patch_low[relevant], patch_high[relevant]
    if not len(patch_low):
        return False
    axes = [
        np.unique(
            np.clip(
                np.r_[low[a], high[a], patch_low[:, a], patch_high[:, a]],
                low[a],
                high[a],
            )
        )
        for a in (0, 1)
    ]
    for x in (axes[0][:-1] + axes[0][1:]) / 2:
        active = (patch_low[:, 0] <= x) & (patch_high[:, 0] >= x)
        for y in (axes[1][:-1] + axes[1][1:]) / 2:
            if not np.any(active & (patch_low[:, 1] <= y) & (patch_high[:, 1] >= y)):
                return False
    return True


def validate_support_placement(target_xyz, size_xyz, support, instances, floor_z):
    """Require sampled continuous support, edge margin, height and empty volume."""
    position = np.asarray(target_xyz, dtype=float)
    size = np.asarray(size_xyz, dtype=float)
    bottom = position[2] - size[2] / 2
    height = round(float(bottom - floor_z), 12)
    if support["category"] not in SUPPORTED_WORKTOP_CATEGORIES:
        raise ValueError("Fetch requires a worktop, not a chair/bench seat")
    if not FETCH_SUPPORT_HEIGHT_BAND_M[0] <= height <= FETCH_SUPPORT_HEIGHT_BAND_M[1]:
        raise ValueError(
            f"Fetch support height {height:.3f} m is outside {FETCH_SUPPORT_HEIGHT_BAND_M}"
        )
    if position[2] + size[2] / 2 - floor_z > FETCH_TARGET_MAX_TOP_M:
        raise ValueError(
            "Fetch target is too high for the conservative production band"
        )
    samples = np.asarray(support["support_samples_xyz_m"], dtype=float).reshape(-1, 3)
    if not len(samples):
        raise ValueError("Fetch support has no measured continuous flat surface")
    offsets = np.linalg.norm(samples[:, :2] - position[:2], axis=1)
    valid = (offsets <= SUPPORT_SAMPLE_POSITION_TOLERANCE_M) & (
        np.abs(samples[:, 2] - bottom) <= SUPPORT_SAMPLE_HEIGHT_TOLERANCE_M
    )
    if not valid.any():
        raise ValueError(
            "Fetch target does not coincide with a measured support sample"
        )
    error = float(offsets[valid].min())
    # Each sampled patch stores its maximum height. Do not union separate
    # height bands: their individual 6 mm flatness bounds would accumulate.
    reference_height = samples[np.flatnonzero(valid)[np.argmin(offsets[valid])], 2]
    coplanar = samples[np.abs(samples[:, 2] - reference_height) <= 1e-9]
    if not _support_patch_covers_footprint(position[:2], size[:2], coplanar):
        raise ValueError(
            "Target footprint and edge margin exceed measured support patch"
        )
    low = position - size / 2
    high = position + size / 2
    # Ignore supporting contacts at the bottom, but never a neighbouring object
    # intersecting the target body. AABB rejection is deliberately conservative.
    low[2] += SUPPORT_SAMPLE_HEIGHT_TOLERANCE_M
    for item in instances:
        if item["prim_path"] == support["prim_path"]:
            continue
        if np.all(high > np.asarray(item["bounds_min_m"])) and np.all(
            low < np.asarray(item["bounds_max_m"])
        ):
            if "/Meshes/Base/" in item["prim_path"]:
                if "surface_bounds_m" not in item:
                    raise ValueError(
                        "Structural target obstruction needs fresh per-face bounds"
                    )
                faces = np.asarray(item["surface_bounds_m"], dtype=float).reshape(
                    -1, 2, 3
                )
                if not (
                    np.all(high > faces[:, 0], axis=1)
                    & np.all(low < faces[:, 1], axis=1)
                ).any():
                    continue
            raise ValueError(
                f"Fetch spawn overlaps neighbouring geometry: {item['prim_path']}"
            )
    return {
        "support_height_m": height,
        "support_sample_error_m": error,
        "support_edge_margin_m": FETCH_SUPPORT_EDGE_MARGIN_M,
    }


def _facing(yaw, start, end, tolerance_rad=math.radians(45)):
    desired = math.atan2(end[1] - start[1], end[0] - start[0])
    return abs((yaw - desired + math.pi) % (2 * math.pi) - math.pi) <= tolerance_rad


def validate_task_layout(
    task,
    kind,
    record,
    meta,
    geometry,
    masks,
    *,
    asset=None,
    contact_profile_path: Path | str | None = None,
    contact_profile_source_root: Path | str | None = None,
):
    """One shared production gate for route rendering and episode compilation."""
    markers = record["markers_world_xy"]
    actors = record["actor_exclusion_centers_world_xy"]
    start_clearance = float(record["start_extra_clearance_m"])
    if start_clearance < MIN_ROUTE_START_EXTRA_CLEARANCE_M:
        raise ValueError(
            f"Production start has only {start_clearance:.3f} m extra clearance; "
            f"require at least {MIN_ROUTE_START_EXTRA_CLEARANCE_M:.2f} m"
        )
    minimum = (
        MIN_FETCH_NAVIGATE_TO_GRASP_LENGTH_M
        if kind == "grasp_task"
        else MIN_INSPECTION_OUTBOUND_LENGTH_M
    )
    if (
        record["start_to_objective_length_m"] < minimum
        or record["objective_to_terminal_length_m"] < MIN_COMPLETION_ROUTE_LENGTH_M
    ):
        raise ValueError("Production route lacks meaningful two-leg extent")
    if not 1 <= len(actors) <= MAX_BASE_NPCS:
        raise ValueError("Production base permits one to three NPCs")
    authored = task["actors"]
    if len(authored) != len(actors) or {a["id"] for a in authored} != set(actors):
        raise ValueError("Authored NPC identities must exactly match route exclusions")
    roles = {a["role"]: a["id"] for a in authored}
    if len(roles) != len(authored):
        raise ValueError("NPC roles must be unique")
    requester = task["requester"]
    recipient = task.get("recipient") or task.get("report_recipient") or requester
    geometry_by_prim = {
        str(item["prim_path"]): item for item in geometry.get("instances", ())
    }
    native_paths: set[str] = set()
    native_ids: set[str] = set()
    for actor in authored:
        source_path = str(actor.get("scene_person_prim_path") or "").strip()
        native_fields = {
            "scene_person_bounds_min_m": actor.get("scene_person_bounds_min_m"),
            "scene_person_bounds_max_m": actor.get("scene_person_bounds_max_m"),
            "public_description": actor.get("public_description"),
            "public_job": actor.get("public_job"),
        }
        if not source_path:
            if any(value is not None for value in native_fields.values()):
                raise ValueError(
                    f"NPC {actor['id']!r} has scene-person fields without an exact prim"
                )
            continue
        if source_path in native_paths:
            raise ValueError(f"Duplicate scene-person binding: {source_path}")
        native_paths.add(source_path)
        native_ids.add(str(actor["id"]))
        instance = geometry_by_prim.get(source_path)
        if instance is None or not is_audited_scene_human_prim(
            source_path,
            category=str(instance.get("category")),
        ):
            raise ValueError(
                f"NPC {actor['id']!r} does not bind an exact person in pinned geometry"
            )
        if (
            not str(native_fields["public_description"] or "").strip()
            or not str(native_fields["public_job"] or "").strip()
        ):
            raise ValueError(
                f"Scene-bound NPC {actor['id']!r} needs public identity semantics"
            )
        for field, geometry_field in (
            ("scene_person_bounds_min_m", "bounds_min_m"),
            ("scene_person_bounds_max_m", "bounds_max_m"),
        ):
            authored_bounds = np.asarray(actor.get(field), dtype=float)
            geometry_bounds = np.asarray(instance[geometry_field], dtype=float)
            if (
                authored_bounds.shape != (3,)
                or np.max(np.abs(authored_bounds - geometry_bounds)) > 0.002
            ):
                raise ValueError(f"Scene-bound NPC {actor['id']!r} has stale {field}")
        center_xy = (
            np.asarray(instance["bounds_min_m"][:2], dtype=float)
            + np.asarray(instance["bounds_max_m"][:2], dtype=float)
        ) * 0.5
        if math.dist(center_xy, actors[str(actor["id"])]) > 0.05:
            raise ValueError(
                f"Scene-bound NPC {actor['id']!r} point is not its geometry center"
            )
    if geometry.get("collection") == "commercial_scenes":
        available_native_people = []
        for instance in geometry.get("instances", ()):
            if (
                not is_audited_scene_human_prim(
                    str(instance.get("prim_path")),
                    category=str(instance.get("category")),
                )
                or str(instance.get("prim_path")) in native_paths
            ):
                continue
            bounds_min = np.asarray(instance.get("bounds_min_m"), dtype=float)
            bounds_max = np.asarray(instance.get("bounds_max_m"), dtype=float)
            if (
                bounds_min.shape != (3,)
                or bounds_max.shape != (3,)
                or bounds_max[2] - bounds_min[2]
                < MIN_INTERACTABLE_SCENE_PERSON_HEIGHT_M
            ):
                continue
            available_native_people.append(
                (
                    str(instance["prim_path"]),
                    (bounds_min[:2] + bounds_max[:2]) * 0.5,
                )
            )
        for actor in authored:
            if str(actor.get("scene_person_prim_path") or "").strip():
                continue
            actor_xy = np.asarray(actors[str(actor["id"])], dtype=float)
            for prim_path, native_xy in available_native_people:
                distance = float(np.linalg.norm(actor_xy - native_xy))
                if distance < MIN_CATALOG_NATIVE_PERSON_SEPARATION_M:
                    raise ValueError(
                        f"Catalog NPC {actor['id']!r} is {distance:.3f} m from "
                        f"an unbound scene-native person at the same semantic "
                        f"location ({prim_path}); bind that exact prim or move "
                        "the catalog role"
                    )
    layout = task["npc_layout"]
    site_human = None
    if layout == "A_return":
        if len(actors) != 1 or recipient != requester:
            raise ValueError("Single-NPC return layout is inconsistent")
    elif layout == "AB":
        site = [a["id"] for a in task["actors"] if a["role"] == "site_contact"]
        if len(actors) != 2 or recipient != requester or len(site) != 1:
            raise ValueError("AB layout requires requester return and one B contact")
        site_human = site[0]
    elif layout == "BC":
        if len(actors) != 2 or recipient == requester:
            raise ValueError(
                "BC layout requires a B requester and a distinct C recipient"
            )
        site_human = requester
    elif layout == "AC":
        if len(actors) != 2 or recipient == requester:
            raise ValueError("AC layout requires distinct requester and recipient")
    elif layout == "ABC":
        recipient_role = "recipient" if kind == "grasp_task" else "report_recipient"
        site_human = roles.get("site_contact")
        if (
            len(actors) != 3
            or len({requester, recipient, site_human}) != 3
            or roles
            != {
                "requester": requester,
                recipient_role: recipient,
                "site_contact": site_human,
            }
        ):
            raise ValueError(
                "ABC layout requires distinct A requester, B contact and C recipient"
            )
    else:
        raise ValueError("Unknown authored NPC layout")
    yaw = {a["id"]: float(a["start_yaw_rad"]) for a in task["actors"]}
    interactions = [
        ("opening", markers["start"], requester),
        ("completion", markers["terminal"], recipient),
    ]
    onsite_mode = str(task.get("onsite_mode") or "inspection")
    if kind == "semantic_task" and onsite_mode not in {"inspection", "inquiry"}:
        raise ValueError("Navigation onsite_mode must be inspection or inquiry")
    if kind != "semantic_task" and "onsite_mode" in task:
        raise ValueError("Only navigation tasks may declare onsite_mode")
    if onsite_mode == "inquiry" and not task.get("site_interaction"):
        raise ValueError("Inquiry navigation requires deterministic onsite dialogue")
    if task.get("site_interaction"):
        SiteInteraction.model_validate(task["site_interaction"])
        if task["site_interaction"]["human_id"] != site_human:
            raise ValueError("Onsite dialogue must bind the authored B contact")
    if site_human is not None:
        interactions.append(("onsite", markers["objective"], site_human))
    for label, stance, human in interactions:
        if human not in actors:
            raise ValueError(f"{label}: interaction names an undeclared NPC")
        if math.dist(stance, actors[human]) > HUMAN_COMMUNICATION_RADIUS_M:
            raise ValueError(f"{label}: interaction exceeds communication range")
        actor = next(value for value in authored if value["id"] == human)
        terminal_clearance_m = 0.0
        if human in native_ids:
            bounds_min = np.asarray(actor["scene_person_bounds_min_m"][:2], dtype=float)
            bounds_max = np.asarray(actor["scene_person_bounds_max_m"][:2], dtype=float)
            terminal_clearance_m = float(
                np.linalg.norm(bounds_max - bounds_min) / 2.0 + 0.03
            )
        if not segment_clear(
            masks["sight"],
            stance,
            actors[human],
            meta,
            terminal_clearance_m=terminal_clearance_m,
        ):
            raise ValueError(f"{label}: interaction is obstructed by scene geometry")
        if human not in native_ids and not _facing(
            yaw[human],
            actors[human],
            stance,
            INITIAL_REQUEST_FACING_TOLERANCE_RAD
            if label == "opening"
            else math.radians(45),
        ):
            raise ValueError(f"{label}: NPC faces away from its interaction stance")
    if not _facing(
        float(task["robot_start_yaw_rad"]),
        markers["start"],
        actors[requester],
        INITIAL_REQUEST_FACING_TOLERANCE_RAD,
    ):
        raise ValueError("Opening robot heading faces away from requester")
    result = {
        "interaction_segments_checked": len(interactions),
        "npc_layout": layout,
        "obstruction_screen": "conservative_mesh_projection",
        "physical_and_rgbd_certification": "not_run",
    }
    result["interaction_distance_review_required"] = [
        {
            "interaction": label,
            "human_id": human,
            "distance_m": round(math.dist(stance, actors[human]), 3),
        }
        for label, stance, human in interactions
        if label != "opening"
        and math.dist(stance, actors[human])
        > PREFERRED_REVIEW_HUMAN_INTERACTION_DISTANCE_M
    ]
    if kind == "grasp_task":
        if asset is None:
            raise ValueError(
                "Certified asset dimensions are required for fetch placement"
            )
        if contact_profile_path is None:
            raise ValueError("Fetch layout authoring requires contact_profile_path")
        profile = load_contact_profile(
            contact_profile_path, source_root=contact_profile_source_root
        )
        target = np.asarray(record["target"]["authored_world_xy"])
        distance = math.dist(markers["objective"], target)
        if (
            not FETCH_STANCE_DISTANCE_BAND_M[0]
            <= distance
            <= FETCH_STANCE_DISTANCE_BAND_M[1]
        ):
            raise ValueError(
                f"Fetch stance distance {distance:.3f} m is outside conservative {FETCH_STANCE_DISTANCE_BAND_M}"
            )
        if not foot_placement_clear(
            masks["feet"], markers["objective"], target, meta, profile=profile
        ):
            raise ValueError(
                "Fetch stance puts the forward feet inside low solid geometry"
            )
        support = next(
            i
            for i in geometry["instances"]
            if i["prim_path"] == task["target_support_source_prim"]
        )
        position = [
            *target,
            task["target_support_top_z_m"] + asset["size_xyz_m"][2] / 2,
        ]
        result["fetch"] = validate_support_placement(
            position,
            asset["size_xyz_m"],
            support,
            geometry["instances"],
            geometry["floor_z_m"],
        )
        result["fetch"]["stance_distance_m"] = distance
        result["fetch"]["oriented_footprint_clear"] = True
        result["fetch"]["robot_contact_profile_sha256"] = digest(
            Path(contact_profile_path).expanduser().resolve()
        )
    return result


def validate_episode_layout(
    episode,
    root,
    *,
    contact_profile_path: Path | str | None = None,
    contact_profile_source_root: Path | str | None = None,
):
    """Revalidate compiled production YAML against its exact authored evidence."""
    sources = episode.world_knowledge["scene_sources"]
    design = json.loads((root / sources["task_design"]).read_text())
    scene = next(
        s
        for s in design["scenes"]
        if s["scene_id"] == episode.world_knowledge["scene_id"]
    )
    kind, task = next(
        (k, t) for k, t in scene["tasks"].items() if t["task_id"] == episode.episode_id
    )
    index = json.loads((root / sources["route_index"]).read_text())
    record = next(s for s in index["scenes"] if s["scene_id"] == scene["scene_id"])[
        "tasks"
    ][kind]
    folder = (
        root
        / "assets/scenes/GRScenes-100"
        / scene["scene_collection"]
        / "scenes"
        / scene["scene_id"]
    )
    meta = json.loads((folder / "bev_meta.json").read_text())
    path = root / sources["layout_geometry"]
    geometry, masks = load_layout_geometry(path, meta, folder / "bev_meta.json")
    if record["layout_geometry_sha256"] != digest(path):
        raise ValueError("Compiled route layout geometry is stale")
    positions = {h.id: list(h.start_position[:2]) for h in episode.humans}
    if positions != record["actor_exclusion_centers_world_xy"]:
        raise ValueError("Compiled NPC positions differ from reviewed layout")
    authored_actors = {actor["id"]: actor for actor in task["actors"]}
    for human in episode.humans:
        if (
            not human.static
            or abs(
                (human.start_yaw - authored_actors[human.id]["start_yaw_rad"] + math.pi)
                % (2 * math.pi)
                - math.pi
            )
            > 0.001
        ):
            raise ValueError(
                "Compiled NPC heading/static state differs from reviewed layout"
            )
    if (
        math.dist(episode.robot.start_position[:2], record["markers_world_xy"]["start"])
        > 0.002
    ):
        raise ValueError("Compiled start differs from reviewed layout")
    w, x, y, z = episode.robot.start_orientation
    robot_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if (
        abs(
            (robot_yaw - task["robot_start_yaw_rad"] + math.pi) % (2 * math.pi)
            - math.pi
        )
        > 0.001
    ):
        raise ValueError("Compiled robot heading differs from reviewed layout")
    if task.get("site_interaction"):
        site = SiteInteraction.model_validate(task["site_interaction"])
        intents = [
            i for i in episode.interaction.intents if i.intent_id == site.intent_id
        ]
        if (
            len(intents) != 1
            or episode.metadata.get("onsite_interaction_intent_id") != site.intent_id
            or intents[0].eligible_human_ids != [site.human_id]
            or intents[0].description != site.description
            or intents[0].answer.kind != "authored_text"
            or intents[0].answer.text != site.answer
        ):
            raise ValueError("Compiled onsite dialogue differs from reviewed task")
    elif episode.metadata.get("onsite_interaction_intent_id"):
        raise ValueError("Compiled episode declares an unauthored onsite dialogue")
    asset = None
    if kind == "grasp_task":
        from tarsphere.bench.certification.certified_asset_pool import (
            certified_asset,
            load_certified_asset_pool,
        )

        pool_path = root / sources["certified_asset_pool"]
        if digest(pool_path) != episode.metadata["certified_asset_pool_sha256"]:
            raise ValueError("Compiled certified asset pool is stale")
        asset = certified_asset(
            load_certified_asset_pool(pool_path), task["scene_adaptation"]["asset_key"]
        )
        if task["scene_adaptation"]["asset_path"] != asset["usd_path"]:
            raise ValueError("Authored asset path differs from the certified pool")
        target = next(
            o
            for o in episode.world_knowledge["objects"]
            if o["object_id"] == task["target_object_id"]
        )
        expected = [
            *record["target"]["authored_world_xy"],
            task["target_support_top_z_m"] + asset["size_xyz_m"][2] / 2,
        ]
        if np.linalg.norm(np.asarray(target["position_xyz"]) - expected) > 0.002:
            raise ValueError("Compiled target differs from support-screened placement")
        validate_runtime_fetch_binding(
            episode.world_knowledge, task, record, asset, expected
        )
    result = validate_task_layout(
        task,
        kind,
        record,
        meta,
        geometry,
        masks,
        asset=asset,
        contact_profile_path=contact_profile_path,
        contact_profile_source_root=contact_profile_source_root,
    )
    if (
        kind == "grasp_task"
        and record["layout_checks"]["fetch"]["robot_contact_profile_sha256"]
        != result["fetch"]["robot_contact_profile_sha256"]
    ):
        raise ValueError("Compiled robot contact profile is stale")
    return result


def validate_runtime_fetch_binding(world, task, record, asset, expected_position):
    """Check the injection/physical fields actually consumed by the harness."""
    adaptation = world["asset_adaptation"]
    instances = adaptation["instances"]
    if len(instances) != 1 or adaptation["hidden_runtime_prims"]:
        raise ValueError(
            "Production fetch requires one target injection without hidden furniture"
        )
    instance = instances[0]
    expected = {
        "object_id": task["target_object_id"],
        "role": "target",
        "asset_key": asset["asset_key"],
        "category": asset["category"],
        "asset_path": asset["usd_path"],
        "asset_meters_per_unit": asset["meters_per_unit"],
        "runtime_prim": task["target_runtime_prim"],
        "size_xyz_m": asset["size_xyz_m"],
        "support_object_id": task["target_support"],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    if any(instance[key] != value for key, value in expected.items()):
        raise ValueError(
            "Runtime target injection differs from the certified authored binding"
        )
    if not np.allclose(
        instance["position_xyz_m"], expected_position, atol=0.002, rtol=0
    ):
        raise ValueError(
            "Runtime target spawn differs from the support-screened placement"
        )
    physical = world["physical_task"]
    grasp = physical["grasp"]
    objective = record["markers_world_xy"]["objective"]
    if (
        physical["target_object_id"] != task["target_object_id"]
        or physical["target_support_object_id"] != task["target_support"]
        or grasp["target_object_id"] != task["target_object_id"]
    ):
        raise ValueError("Physical grasp binds a different target/support")
    if any(
        not np.allclose(grasp[key], objective, atol=0.002, rtol=0)
        for key in ("navigation_approach_xy", "robot_stance_xy", "retreat_xy")
    ):
        raise ValueError("Physical grasp stance differs from the screened B pose")
    desired = math.atan2(
        expected_position[1] - objective[1], expected_position[0] - objective[0]
    )
    if (
        abs((grasp["robot_yaw_rad"] - desired + math.pi) % (2 * math.pi) - math.pi)
        > 0.001
    ):
        raise ValueError("Physical grasp yaw does not face the screened target")
