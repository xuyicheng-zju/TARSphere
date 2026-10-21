"""Resolve an authored HC conversation detour without moving Base task loci."""

from copy import deepcopy
import json
import math
from pathlib import Path

from tarsphere.harness.perception.backends.gt_bev import OfficialOccupancyBevMap
from tarsphere.harness.planner.planner import Planner
from tarsphere.bench.certification.scene import official_runtime_navigation_mask
from tarsphere.bench.production_layout import load_layout_geometry, segment_clear
from tarsphere.bench.spatial.physical_task.navigation_evidence import path_length
from tarsphere.contracts.human_interaction import HUMAN_COMMUNICATION_RADIUS_M
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    HUMAN_BODY_RADIUS_M,
)

CONVERSATION_POINT_ID = "coordination_stance"
RETURN_POINT_ID = "coordination_return"
ROOT = Path(__file__).resolve().parents[3]


def require_conversation_sight(data: dict, human: dict, position_xy, *, root=ROOT):
    """Static mesh-projection screen, not a runtime RGB-D visibility certificate."""
    if not all(math.isfinite(float(x)) for x in position_xy):
        raise ValueError("Coordination stance must be finite")
    if (
        math.dist(position_xy, human["start_position"][:2])
        > HUMAN_COMMUNICATION_RADIUS_M
    ):
        raise ValueError("Coordination stance is outside communication range")
    world = data["world_knowledge"]
    projection = root / "assets" / world["scene_map"]["projection"]
    meta = json.loads(projection.read_text())
    _, masks = load_layout_geometry(
        root / world["scene_sources"]["layout_geometry"], meta, projection, root=root
    )
    terminal = 0.0
    if human.get("scene_person_prim_path"):
        lo, hi = human["scene_person_bounds_min_m"], human["scene_person_bounds_max_m"]
        terminal = math.hypot(hi[0] - lo[0], hi[1] - lo[1]) / 2
    if not segment_clear(
        masks["sight"],
        position_xy,
        human["start_position"][:2],
        meta,
        terminal_clearance_m=terminal,
    ):
        raise ValueError(
            f"Coordination stance has an obstructed sight ray to {human['id']}"
        )


def resolve_coordination_detour(
    data: dict, human: dict, anchor_id: str, position_xy, *, root=ROOT
):
    from tarsphere.bench.authoring import (
        CandidateRouteLeg,
        _resolved_leg_path,
        _planner_path_sha256,
        sha256_file,
    )

    require_conversation_sight(data, human, position_xy, root=root)
    route = deepcopy(data["world_knowledge"]["episode_route"])
    leg_name = (
        "navigate_to_target"
        if anchor_id == "task_start"
        else "transport_to_destination"
    )
    points = route[leg_name]["points"]
    insert_at = 1 if anchor_id in ("task_start", "task_target") else len(points) - 1
    if any(
        p["point_id"] == CONVERSATION_POINT_ID
        for leg in ("navigate_to_target", "transport_to_destination")
        for p in route[leg]["points"]
    ):
        raise ValueError("Base route already contains a coordination stance")
    points.insert(
        insert_at,
        {
            "point_id": CONVERSATION_POINT_ID,
            "position_xy": list(position_xy),
            "semantic_anchor": f"Reachable conversation stance facing {human['id']}",
            "anchor_kind": "human_position",
            "anchor_id": human["id"],
            "authored_anchor_xy": human["start_position"][:2],
            "snap_residual_m": math.dist(position_xy, human["start_position"][:2]),
        },
    )
    if anchor_id == "task_target":
        # Onsite coordination can require asking the original source or
        # inspecting the target after the contribution. Show that return,
        # rather than a shortcut from the contributor to the final recipient.
        return_point = deepcopy(points[0])
        return_point["point_id"] = RETURN_POINT_ID
        points.insert(insert_at + 1, return_point)
    occupancy_path = root / "assets" / data["world_knowledge"]["scene_map"]["occupancy"]
    occupancy = OfficialOccupancyBevMap(occupancy_path=occupancy_path)
    obstacles = tuple(
        (*h["start_position"][:2], HUMAN_BODY_RADIUS_M)
        for h in data["humans"]
        if h.get("static") and not h.get("scene_person_prim_path")
    )
    planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        prepared_navigable_mask=official_runtime_navigation_mask(
            occupancy_path,
            clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
            static_obstacles=obstacles,
        ),
        cache_static_nav=True,
    )
    paths = []
    for leg in ("navigate_to_target", "transport_to_destination"):
        path = _resolved_leg_path(
            planner,
            CandidateRouteLeg.model_validate(route[leg]),
            owner=Path(data["episode_id"]),
            leg_name=leg,
        )
        route[leg]["optimal_navigation_length_m"] = path_length(path)
        paths.append(path)
    route.update(
        occupancy_sha256=sha256_file(occupancy_path),
        planner_path_sha256=_planner_path_sha256(*paths),
        collision_cell_count=0,
        optimal_navigation_length_m=sum(path_length(path) for path in paths),
    )
    data["world_knowledge"]["episode_route"] = route


def validate_coordination_contacts(data: dict, *, root=ROOT):
    """Check the actual contribution receipt stop, including reused A/B/C points.

    An explicit detour is not the only way to contact a contributor. Binding a
    receipt to an existing task point requires the same range and sight screen.
    Later collaboration work/motion is not evaluated at the initial Human pose.
    """
    supporter = data["evaluation"]["task_profile"]["supporting_human_id"]
    human = next(h for h in data["humans"] if h["id"] == supporter)
    route = data["world_knowledge"]["episode_route"]
    points = {
        p["point_id"]: p
        for leg in ("navigate_to_target", "transport_to_destination")
        for p in route[leg]["points"]
    }
    stops = {s["milestone_id"]: s["route_point_id"] for s in route["milestone_stops"]}
    contacts = []
    for node in data["evaluation"]["milestones"]:
        outcome = node.get("completion") or node.get("reaction")
        if (
            outcome
            and outcome["evaluator"] == "information_received"
            and outcome["ground_truth"].get("human_id") == supporter
            and outcome["ground_truth"].get("intent_id") == "coordination_contribution"
        ):
            contacts.append(node["milestone_id"])
    if len(contacts) != 1:
        raise ValueError("HC requires one bound contribution receipt contact")
    stance = points[stops[contacts[0]]]
    require_conversation_sight(data, human, stance["position_xy"], root=root)
    explicit = [p for p in points.values() if p["point_id"] == CONVERSATION_POINT_ID]
    if explicit and stance["point_id"] != CONVERSATION_POINT_ID:
        raise ValueError(
            "HC contribution receipt must use the declared conversation stance"
        )


def validate_coordination_route_pair(base: dict, variant: dict):
    """Allow only one contributor visit; Base anchors and route contracts stay fixed."""
    left = deepcopy(base["world_knowledge"]["episode_route"])
    right = deepcopy(variant["world_knowledge"]["episode_route"])
    left.pop("milestone_stops")
    right.pop("milestone_stops")
    additions = [
        (leg, p)
        for leg in ("navigate_to_target", "transport_to_destination")
        for p in right[leg]["points"]
        if p["point_id"] == CONVERSATION_POINT_ID
    ]
    if not additions:
        if left != right:
            raise ValueError(
                "HC changed a Base Robot route without a conversation stance"
            )
        return
    if len(additions) != 1:
        raise ValueError("HC requires one unique conversation stance")
    leg, point = additions[0]
    pattern = variant["evaluation"]["task_profile"]["coordination_pattern"]
    expected_leg = {
        "enroute_information": "navigate_to_target",
        "onsite_information": "transport_to_destination",
        "delivery_information": "transport_to_destination",
    }.get(pattern)
    if expected_leg and leg != expected_leg:
        raise ValueError("HC conversation visit occurs in the wrong task phase")
    point_index = right[leg]["points"].index(point)
    if point_index in (0, len(right[leg]["points"]) - 1):
        raise ValueError("HC conversation visit cannot replace a Base endpoint")
    supporter = variant["evaluation"]["task_profile"]["supporting_human_id"]
    human = next(h for h in variant["humans"] if h["id"] == supporter)
    if (
        point.get("anchor_kind") != "human_position"
        or point.get("anchor_id") != supporter
    ):
        raise ValueError("HC conversation stance must bind its contributor")
    if point.get("authored_anchor_xy") != human["start_position"][:2]:
        raise ValueError("HC conversation stance has stale Human coordinates")
    returns = [
        (part, p)
        for part in ("navigate_to_target", "transport_to_destination")
        for p in right[part]["points"]
        if p["point_id"] == RETURN_POINT_ID
    ]
    if (
        leg == "transport_to_destination"
        and pattern != "completion_collaboration"
        and not returns
    ):
        raise ValueError("HC onsite visit must return to the unchanged task stance")
    if returns:
        if len(returns) != 1 or leg != "transport_to_destination":
            raise ValueError("HC onsite return must occur once on the return leg")
        expected = deepcopy(left[leg]["points"][0])
        expected["point_id"] = RETURN_POINT_ID
        if returns[0] != (leg, expected) or right[leg]["points"][1:3] != [
            point,
            expected,
        ]:
            raise ValueError("HC onsite visit must return to the unchanged task stance")
        right[leg]["points"] = [
            p for p in right[leg]["points"] if p["point_id"] != RETURN_POINT_ID
        ]
    right[leg]["points"] = [
        p for p in right[leg]["points"] if p["point_id"] != CONVERSATION_POINT_ID
    ]
    for route in (left, right):
        for field in ("optimal_navigation_length_m", "planner_path_sha256"):
            route.pop(field)
        for part in ("navigate_to_target", "transport_to_destination"):
            route[part].pop("optimal_navigation_length_m", None)
    if left != right:
        raise ValueError("HC changed a Base route anchor or planning contract")
