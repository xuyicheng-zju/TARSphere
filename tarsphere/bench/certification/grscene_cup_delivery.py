"""Current physical-task contract for the shared real-GRScene cup delivery."""

from __future__ import annotations

import math

from tarsphere.bench.compilation.episode import (
    yaw_quaternion,
)
from tarsphere.bench.compilation.schema import (
    CompiledPhysicalTask,
    EpisodeObject,
)
from tarsphere.bench.grounding import resolve_scene
from tarsphere.bench.grounding.schema import (
    GroundedObject,
    GroundedScene,
)
from tarsphere.bench.spatial.episode_route.schema import EpisodeRoute
from tarsphere.bench.spatial.physical_task.schema import (
    GraspAction,
    NavigationPoint,
    NavigationRoute,
    PlaceAction,
)
from tarsphere.bench.schema import (
    ActionSpec,
    TARSphereEpisode,
    RobotSpec,
    RuntimeSpec,
    SceneSpec,
    DiagnosticCheck,
    DiagnosticSpec,
)
from tarsphere.infra.control.physical_contracts import (
    G1_PLACEMENT_RELATION_TOLERANCE_M,
)


SCENE_ID = "MVUCSQAKTKJ5EAABAAAAADA8_usd"
TARGET_OBJECT_ID = "cup/model_750dbb4f5f7e8803032e7cc56746712e_0"
SOURCE_LOCATION_OBJECT_ID = "table/model_ce0a15c096ad69d94dcfa1f76e6434e1_0"
SOURCE_SUPPORT_OBJECT_ID = "plate/model_260800e518042b1b7b31118609c55a93_0"
DESTINATION_OBJECT_ID = "table/model_5ac32b5baaa93b760c97d641f8d5c5d3_0"

TARGET_NAME = "white round cup with a small side handle"
SOURCE_LOCATION_NAME = "round glass-top table"
SOURCE_SUPPORT_NAME = "white saucer"
DESTINATION_NAME = "rectangular white table"

ROBOT_START_XY = (5.964625, -4.030573)
GRASP_APPROACH_XY = (2.076718330383301, -2.4333887100219727)
GRASP_STANCE_XY = GRASP_APPROACH_XY
PLACE_APPROACH_XY = (0.18830664455890656, -5.4888715744018555)
PLACE_STANCE_XY = PLACE_APPROACH_XY
PLACE_SURFACE_XYZ = (-0.23, -5.502760410308838, 0.532252)
# Clear destination patch certified for the shared Mobile Franka executor.
PLACE_PATCH_SIZE_XY = (0.50, 0.50)

# Offline G1-footprint A* output from the official scene occupancy. Runtime
# policies consume these frozen points and never query occupancy or replan
# from hidden scene geometry.
NAVIGATE_TO_TARGET_XY = (
    ROBOT_START_XY,
    (5.964624881744385, -2.3778343200683594),
    (5.936854362487793, -2.2389488220214844),
    (5.062075138092041, -0.9473126530647278),
    (4.964877605438232, -0.863981306552887),
    (4.72882604598999, -0.8223156332969666),
    (3.5624542236328125, -0.8223156332969666),
    (3.423600435256958, -0.8500927686691284),
    (3.3125174045562744, -0.9334241151809692),
    (3.2153196334838867, -1.0584211349487305),
    (3.159778118133545, -1.2667495012283325),
    (2.9792680740356445, -1.5306322574615479),
    GRASP_APPROACH_XY,
)
TRANSPORT_TO_DESTINATION_XY = (
    GRASP_APPROACH_XY,
    (1.1602833271026611, -3.9472415447235107),
    (1.1602833271026611, -4.336121082305908),
    (1.1186271905899048, -4.558338165283203),
    PLACE_APPROACH_XY,
)


def _scene_objects(
    scene: GroundedScene,
) -> tuple[
    GroundedObject,
    GroundedObject,
    GroundedObject,
    GroundedObject,
]:
    objects = {item.object_id: item for item in scene.objects}
    selected = tuple(
        objects[object_id]
        for object_id in (
            TARGET_OBJECT_ID,
            SOURCE_SUPPORT_OBJECT_ID,
            SOURCE_LOCATION_OBJECT_ID,
            DESTINATION_OBJECT_ID,
        )
    )
    target, source_support, source_location, destination = selected
    if target.interaction != "pickable":
        raise RuntimeError(f"Physical-task target is not pickable: {target.object_id}")
    if (
        target.room_id,
        source_support.room_id,
        source_location.room_id,
        destination.room_id,
    ) != ("R02", "R02", "R02", "R03"):
        raise RuntimeError(
            "Physical-task object rooms disagree with the grounded scene: "
            f"{target.room_id}, {source_support.room_id}, "
            f"{source_location.room_id}, {destination.room_id}"
        )
    return target, source_support, source_location, destination


def _episode_object(item: GroundedObject, *, name: str) -> EpisodeObject:
    if item.runtime_prim is None:
        raise RuntimeError(
            f"Physical-task object has no runtime prim: {item.object_id}"
        )
    return EpisodeObject(
        object_id=item.object_id,
        name=name,
        category=item.category,
        room_id=item.room_id,
        position_xyz=item.position_xyz,
        bounds_min_xyz=item.bounds_min_xyz,
        bounds_max_xyz=item.bounds_max_xyz,
        runtime_prim=item.runtime_prim,
        interaction=item.interaction,
    )


def _route(
    points: tuple[tuple[float, float], ...],
    *,
    prefix: str,
) -> NavigationRoute:
    return NavigationRoute(
        points=tuple(
            NavigationPoint(
                point_id=(
                    "robot_start"
                    if prefix == "unladen" and index == 0
                    else "grasp_approach"
                    if prefix == "unladen" and index == len(points) - 1
                    else "grasp_retreat"
                    if prefix == "loaded" and index == 0
                    else "destination_approach"
                    if prefix == "loaded" and index == len(points) - 1
                    else f"{prefix}_{index:02d}"
                ),
                position_xy=point,
            )
            for index, point in enumerate(points)
        )
    )


def _path_length(points: tuple[tuple[float, float], ...]) -> float:
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def _physical_task(
    target: GroundedObject,
    destination: GroundedObject,
) -> CompiledPhysicalTask:
    grasp_yaw = math.atan2(
        target.position_xyz[1] - GRASP_STANCE_XY[1],
        target.position_xyz[0] - GRASP_STANCE_XY[0],
    )
    place_yaw = math.atan2(
        PLACE_SURFACE_XYZ[1] - PLACE_STANCE_XY[1],
        PLACE_SURFACE_XYZ[0] - PLACE_STANCE_XY[0],
    )
    return CompiledPhysicalTask(
        robot_start_yaw_rad=math.pi / 2.0,
        target_object_id=target.object_id,
        target_support_object_id=SOURCE_SUPPORT_OBJECT_ID,
        initial_destination_object_id=destination.object_id,
        final_destination_object_id=destination.object_id,
        goal_relation="on",
        grasp=GraspAction(
            target_object_id=target.object_id,
            navigation_approach_xy=GRASP_APPROACH_XY,
            robot_stance_xy=GRASP_STANCE_XY,
            robot_yaw_rad=grasp_yaw,
            retreat_xy=GRASP_APPROACH_XY,
        ),
        place=PlaceAction(
            destination_object_id=destination.object_id,
            navigation_approach_xy=PLACE_APPROACH_XY,
            robot_stance_xy=PLACE_STANCE_XY,
            robot_yaw_rad=place_yaw,
            support_surface_xyz=PLACE_SURFACE_XYZ,
        ),
    )


def _episode_route() -> EpisodeRoute:
    return EpisodeRoute(
        navigate_to_target=_route(NAVIGATE_TO_TARGET_XY, prefix="unladen"),
        transport_to_destination=_route(
            TRANSPORT_TO_DESTINATION_XY,
            prefix="loaded",
        ),
        milestone_stops=(),
        optimal_navigation_length_m=(
            _path_length(NAVIGATE_TO_TARGET_XY)
            + _path_length(TRANSPORT_TO_DESTINATION_XY)
        ),
    )


def create_grscene_cup_delivery_episode() -> TARSphereEpisode:
    """Create one Human-free task shared by deterministic and model agents."""

    scene = resolve_scene(SCENE_ID)
    target, source_support, source_location, destination = _scene_objects(scene)
    physical_task = _physical_task(target, destination)
    episode_route = _episode_route()
    objects = (
        _episode_object(target, name=TARGET_NAME),
        _episode_object(source_support, name=SOURCE_SUPPORT_NAME),
        _episode_object(source_location, name=SOURCE_LOCATION_NAME),
        _episode_object(destination, name=DESTINATION_NAME),
    )
    return TARSphereEpisode(
        episode_id="grscene_shared_cup_delivery",
        title="Shared Real-GRScene Cup Delivery",
        description=(
            "Human-free end-to-end certification of the current navigation, "
            "grasp, transport, and support-aware placement harness."
        ),
        system_prompt=(
            f"Pick up the {TARGET_NAME} from the {SOURCE_LOCATION_NAME} in "
            "R02 / Living Room 01. Carry it to R03 / Dinning Room 01 and "
            f"place it stably on the {DESTINATION_NAME}. Report done only "
            "after the cup has been released and remains supported."
        ),
        scene=SceneSpec(
            asset_path=scene.sources.scene_usd,
            scale=scene.scene_scale,
        ),
        robot=RobotSpec(
            id=0,
            name="robot",
            model="G1Robot",
            start_position=(*ROBOT_START_XY, 0.8),
            start_orientation=yaw_quaternion(physical_task.robot_start_yaw_rad),
        ),
        humans=[],
        agent_action_space=ActionSpec(
            cognitive=["wait", "tilt_camera"],
            communicative=["report_done"],
            physical=[
                "move_body",
                "crouch",
                "stand_up",
                "move_end_effector",
                "grasp_object",
            ],
        ),
        diagnostics=DiagnosticSpec(
            description="The cup is stably supported by the destination table.",
            checks=[
                DiagnosticCheck(
                    type="object_supported_by",
                    params={
                        "object_id": target.object_id,
                        "destination_object_id": destination.object_id,
                        "distance": G1_PLACEMENT_RELATION_TOLERANCE_M,
                    },
                ),
                DiagnosticCheck(type="agent_reported_done"),
            ],
        ),
        runtime=RuntimeSpec(
            max_decision_steps=None,
            max_physics_steps=None,
        ),
        world_knowledge={
            "scene_id": scene.scene_id,
            "scene_sources": scene.sources.model_dump(mode="python"),
            "scene_map": scene.map.model_dump(mode="python"),
            "physical_task": physical_task.model_dump(mode="python"),
            "episode_route": episode_route.model_dump(mode="python"),
            "objects": [item.model_dump(mode="python") for item in objects],
            # The diagnostic has no Humans, so movement anchors stay empty.
            # Navigation is bound directly to episode_route point positions;
            # no region annotation participates in execution or evaluation.
            "anchors": {},
        },
        metadata={
            "validation_scope": "complete_physical_task_harness",
            "task_contract": "navigate_grasp_transport_place",
        },
    )


__all__ = [
    "DESTINATION_NAME",
    "DESTINATION_OBJECT_ID",
    "GRASP_APPROACH_XY",
    "NAVIGATE_TO_TARGET_XY",
    "PLACE_APPROACH_XY",
    "PLACE_PATCH_SIZE_XY",
    "PLACE_SURFACE_XYZ",
    "ROBOT_START_XY",
    "SCENE_ID",
    "SOURCE_SUPPORT_OBJECT_ID",
    "SOURCE_LOCATION_OBJECT_ID",
    "TARGET_NAME",
    "TARGET_OBJECT_ID",
    "TRANSPORT_TO_DESTINATION_XY",
    "create_grscene_cup_delivery_episode",
]
