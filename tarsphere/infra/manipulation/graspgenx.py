"""GraspGenX poses for the native Unitree Dex3-1 hand.

The evaluated Agent names a visible object from RGB.  Synchronized depth and
the SAM3 mask remain private to this manipulation skill and are sent to one
GraspGenX inference server.  GraspGenX returns full gripper poses in the camera
frame; this module never invents a geometric fallback when the server, depth,
or target region is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


# Official ``unitree_g1`` conditioning from NVIDIA's gripper-descriptions
# release.  GraspGenX defines +Z as approach and +X as finger closure.
UNITREE_G1_SWEEP_VOLUME = {
    "extents_open": np.asarray([0.10, 0.06, 0.04], dtype=np.float32),
    "offset_open": np.asarray([0.0, 0.0, 0.07], dtype=np.float32),
    "extents_mid": np.asarray([0.04, 0.06, 0.04], dtype=np.float32),
    "offset_mid": np.asarray([0.007, 0.0, 0.06], dtype=np.float32),
    "gripper_type": 2,
    "fingertip_depth": 0.07,
}

# Enter from one physical fingertip depth behind the generated closing pose.
# The former 0.12 m retreat was larger than the complete Dex3 finger volume
# and could move an otherwise reachable low grasp into a near-body singularity.
UNITREE_G1_PREGRASP_STANDOFF_M = float(UNITREE_G1_SWEEP_VOLUME["fingertip_depth"])

# GPU kernels can differ in their final floating-point digits even after all
# random sources are seeded. Four decimal places preserve sub-millimetre pose
# precision while making the inference boundary reproducible.
GRASPGENX_REPRODUCIBLE_DECIMALS = 4

# GraspGenX poses use the standardized ``unitree_g1`` gripper frame, not a
# physical Dex3 palm frame. The released right-hand descriptor fixes its palm
# at xyz=(0.07, 0.0, 0.02), rpy=(pi/2, 0, pi). Native G1 left-hand geometry is
# the right hand reflected across the palm x-z plane. A reflection is not a
# legal robot pose, so the corresponding standardized-frame lateral axis is
# reflected as well. The two reflections preserve GraspGenX's approach,
# closing axis, and symmetric sweep volume while keeping both rotations in
# SO(3).
UNITREE_G1_GRIPPER_FROM_DEX3_PALM = {
    "right": np.asarray(
        (
            (-1.0, 0.0, 0.0, 0.07),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 0.02),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    ),
    "left": np.asarray(
        (
            (-1.0, 0.0, 0.0, 0.07),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, -1.0, 0.0, 0.02),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    ),
}


class NoGraspPoseError(RuntimeError):
    """The current target geometry contains no usable GraspGenX pose."""


@dataclass(frozen=True)
class GraspGenXSettings:
    """Connection and inference limits for the GraspGenX service.

    Attributes:
        host: Hostname of the dedicated GraspGenX inference process.
        port: ZMQ request port exposed by that process.
        request_timeout_ms: Maximum time allowed for one inference request.
        maximum_depth_m: Furthest depth sample admitted to grasp generation.
        minimum_target_points: Minimum valid target pixels required by the
            model; smaller masks are not geometrically trustworthy.
        generated_pose_count: Number of raw poses requested from GraspGenX.
            Matches GraspGenX ``num_grasps`` default (200).
        returned_pose_count: Highest-confidence poses retained for robot
            reachability planning. Matches GraspGenX ``topk_num_grasps``
            default (100) on the official client, ZMQ ``infer`` handler,
            and demo ``--return_topk --topk_num_grasps 100``.
        minimum_confidence: Lowest GraspGenX discriminator score retained.
        inference_seed: Seed that makes repeated inference on the same geometry
            produce the same ordered grasp candidates.
    """

    host: str = "127.0.0.1"
    port: int = 5556
    request_timeout_ms: int = 180_000
    maximum_depth_m: float = 1.50
    minimum_target_points: int = 100
    generated_pose_count: int = 200
    returned_pose_count: int = 100
    minimum_confidence: float = -1.0
    inference_seed: int = 0

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError("GraspGenX host cannot be empty.")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("GraspGenX port must lie in [1, 65535].")
        if int(self.request_timeout_ms) <= 0:
            raise ValueError("GraspGenX request timeout must be positive.")
        if not math.isfinite(self.maximum_depth_m) or self.maximum_depth_m <= 0:
            raise ValueError("GraspGenX maximum depth must be finite and positive.")
        if int(self.minimum_target_points) < 32:
            raise ValueError("GraspGenX requires at least 32 target points.")
        if int(self.generated_pose_count) < 1:
            raise ValueError("GraspGenX generated pose count must be positive.")
        if not 1 <= int(self.returned_pose_count) <= int(self.generated_pose_count):
            raise ValueError(
                "GraspGenX returned pose count must be positive and no larger "
                "than the generated pose count."
            )
        if not math.isfinite(self.minimum_confidence):
            raise ValueError("GraspGenX minimum confidence must be finite.")
        if int(self.inference_seed) < 0:
            raise ValueError("GraspGenX inference seed must be non-negative.")


@dataclass(frozen=True)
class GraspPoseCandidate:
    """One ranked GraspGenX gripper pose in the synchronized camera frame.

    Attributes:
        confidence: GraspGenX discriminator score; larger values rank first.
        camera_from_gripper: Row-major homogeneous transform from the
            GraspGenX gripper frame to the camera frame.  In that gripper
            frame +Z is approach and +X is the three-finger closing axis.
    """

    confidence: float
    camera_from_gripper: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.confidence)):
            raise ValueError("A GraspGenX confidence must be finite.")
        transform = np.asarray(self.camera_from_gripper, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("A GraspGenX pose must be a finite 4x4 transform.")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=2e-3):
            raise ValueError("A GraspGenX pose must use homogeneous coordinates.")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError("A GraspGenX pose rotation must be orthonormal.")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
            raise ValueError("A GraspGenX pose rotation must be right-handed.")

    @property
    def approach_axis_camera(self) -> np.ndarray:
        """Unit approach direction in camera coordinates."""

        return np.asarray(self.camera_from_gripper, dtype=np.float64)[:3, 2]

    @property
    def closing_axis_camera(self) -> np.ndarray:
        """Unit three-finger closing direction in camera coordinates."""

        return np.asarray(self.camera_from_gripper, dtype=np.float64)[:3, 0]

    @property
    def sweep_center_camera_m(self) -> np.ndarray:
        """Centre of the open finger volume in camera metres."""

        transform = np.asarray(self.camera_from_gripper, dtype=np.float64)
        offset = np.asarray(
            UNITREE_G1_SWEEP_VOLUME["offset_open"],
            dtype=np.float64,
        )
        return transform[:3, :3] @ offset + transform[:3, 3]

    def palm_pose_camera(
        self,
        *,
        hand: str,
        standoff_m: float = 0.0,
    ) -> np.ndarray:
        """Return one native Dex3 palm pose in camera space.

        ``standoff_m`` backs the complete gripper away along its standardized
        +Z approach axis before composing the hand-specific palm mount. Both
        hands therefore enter along the identical generated approach line;
        only the palm chirality differs.
        """

        normalized_hand = str(hand).strip().lower()
        if normalized_hand not in UNITREE_G1_GRIPPER_FROM_DEX3_PALM:
            raise ValueError("A Dex3 palm hand must be left or right.")
        distance = float(standoff_m)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("GraspGenX standoff must be finite and non-negative.")
        camera_from_gripper = np.asarray(
            self.camera_from_gripper,
            dtype=np.float64,
        ).copy()
        camera_from_gripper[:3, 3] -= distance * camera_from_gripper[:3, 2]
        return camera_from_gripper @ UNITREE_G1_GRIPPER_FROM_DEX3_PALM[normalized_hand]


class GraspGenXClient:
    """Thin ZMQ client for one separately hosted GraspGenX model."""

    def __init__(self, *, settings: GraspGenXSettings) -> None:
        self.settings = settings
        self._context: Any = None
        self._socket: Any = None
        self._metadata: dict[str, Any] | None = None

    @property
    def address(self) -> str:
        """ZMQ address of the configured GraspGenX service."""

        return f"tcp://{self.settings.host}:{self.settings.port}"

    def close(self) -> None:
        """Close the request socket without terminating the shared context."""

        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def require_ready(self) -> dict[str, Any]:
        """Fail loudly unless the configured server is alive and has models."""

        health = self._request({"action": "health"})
        metadata = self._request({"action": "metadata"})
        if not isinstance(health, dict) or not health.get("status") == "ok":
            raise RuntimeError(
                f"GraspGenX at {self.address} is not healthy: {health!r}"
            )
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"GraspGenX at {self.address} returned invalid metadata."
            )
        if metadata.get("default_gripper") != "unitree_g1":
            raise RuntimeError(
                f"GraspGenX at {self.address} has the wrong default gripper: "
                f"{metadata.get('default_gripper')!r}"
            )
        required_actions = {"health", "metadata", "infer_scene_depth"}
        missing_actions = required_actions - set(metadata.get("actions") or ())
        if missing_actions:
            raise RuntimeError(
                f"GraspGenX at {self.address} is missing required actions: "
                f"{sorted(missing_actions)}"
            )
        self._metadata = metadata
        return metadata

    def propose(
        self,
        *,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        target_mask: np.ndarray,
    ) -> tuple[GraspPoseCandidate, ...]:
        """Generate target-conditioned poses from one synchronized RGB-D view."""

        depth = np.asarray(depth_m, dtype=np.float32)
        mask = np.asarray(target_mask, dtype=bool)
        camera_matrix = np.asarray(intrinsics, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError(f"GraspGenX depth must be HxW; got {depth.shape!r}.")
        if mask.shape != depth.shape:
            raise ValueError(
                "GraspGenX target mask and depth must describe the same frame: "
                f"mask={mask.shape!r}, depth={depth.shape!r}."
            )
        if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
            raise ValueError("GraspGenX intrinsics must be a finite 3x3 matrix.")
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= float(self.settings.maximum_depth_m))
        )
        target_valid = valid & mask
        target_count = int(np.count_nonzero(target_valid))
        if target_count < int(self.settings.minimum_target_points):
            raise NoGraspPoseError(
                "GraspGenX target region has only "
                f"{target_count} valid depth points; requires "
                f"{self.settings.minimum_target_points}. Move closer or expose "
                "more of the object before grasping."
            )
        bounded_depth = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
        instance_mask = target_valid.astype(np.int32, copy=False)
        response = self._request(
            {
                "action": "infer_scene_depth",
                "depth": bounded_depth,
                "intrinsics": camera_matrix,
                "instance_mask": instance_mask,
                "sweep_volume_params": dict(UNITREE_G1_SWEEP_VOLUME),
                "planner": "graspmoe",
                "min_object_points": int(self.settings.minimum_target_points),
                "num_grasps": int(self.settings.generated_pose_count),
                "seed": int(self.settings.inference_seed),
            }
        )
        candidates = _target_candidates(
            response,
            minimum_confidence=float(self.settings.minimum_confidence),
            maximum_count=int(self.settings.returned_pose_count),
        )
        if not candidates:
            raise NoGraspPoseError(
                "GraspGenX returned no pose for the current target geometry."
            )
        return candidates

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute one msgpack request and reset a timed-out REQ socket."""

        try:
            import msgpack
            import msgpack_numpy
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "GraspGenX client dependencies are missing. Install the TARSphere "
                "`grasp` extra before constructing GraspGenX."
            ) from exc
        msgpack_numpy.patch()
        if self._socket is None:
            self._context = zmq.Context.instance()
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, int(self.settings.request_timeout_ms))
            socket.setsockopt(zmq.SNDTIMEO, int(self.settings.request_timeout_ms))
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self.address)
            self._socket = socket
        try:
            self._socket.send(msgpack.packb(payload, use_bin_type=True))
            packed = self._socket.recv()
        except zmq.error.Again as exc:
            self.close()
            raise TimeoutError(
                f"GraspGenX at {self.address} did not respond within "
                f"{self.settings.request_timeout_ms} ms."
            ) from exc
        response = msgpack.unpackb(packed, raw=False)
        if not isinstance(response, dict):
            raise RuntimeError(
                f"GraspGenX at {self.address} returned a non-object response."
            )
        if "error" in response:
            raise RuntimeError(f"GraspGenX server error: {response['error']}")
        return response


def grasp_pose_payload(candidate: GraspPoseCandidate) -> dict[str, Any]:
    """Serialize one candidate into action metadata owned by the Agent step."""

    return {
        "confidence": float(candidate.confidence),
        "camera_from_gripper": [
            [float(value) for value in row] for row in candidate.camera_from_gripper
        ],
    }


def grasp_pose_from_payload(payload: Any) -> GraspPoseCandidate:
    """Validate one candidate after it crosses the action boundary."""

    if not isinstance(payload, dict):
        raise ValueError("A serialized GraspGenX pose must be an object.")
    if set(payload) != {"confidence", "camera_from_gripper"}:
        raise ValueError(
            "A serialized GraspGenX pose requires exactly `confidence` and "
            "`camera_from_gripper`."
        )
    pose = np.asarray(payload["camera_from_gripper"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera_from_gripper must be 4x4.")
    return GraspPoseCandidate(
        confidence=float(payload["confidence"]),
        camera_from_gripper=tuple(tuple(float(value) for value in row) for row in pose),
    )


def _target_candidates(
    response: dict[str, Any],
    *,
    minimum_confidence: float,
    maximum_count: int,
) -> tuple[GraspPoseCandidate, ...]:
    """Read instance id 1 from GraspGenX's parallel-list scene response."""

    instance_ids = np.asarray(response.get("instance_ids", [])).reshape(-1)
    poses_by_instance = response.get("grasps")
    scores_by_instance = response.get("confidences")
    if not isinstance(poses_by_instance, list) or not isinstance(
        scores_by_instance, list
    ):
        raise RuntimeError("GraspGenX scene response lacks grasp arrays.")
    for index, instance_id in enumerate(instance_ids.tolist()):
        if int(instance_id) != 1:
            continue
        poses = np.round(
            np.asarray(poses_by_instance[index], dtype=np.float64).reshape(-1, 4, 4),
            decimals=GRASPGENX_REPRODUCIBLE_DECIMALS,
        )
        scores = np.round(
            np.asarray(scores_by_instance[index], dtype=np.float64).reshape(-1),
            decimals=GRASPGENX_REPRODUCIBLE_DECIMALS,
        )
        if len(poses) != len(scores):
            raise RuntimeError("GraspGenX returned mismatched poses and confidences.")
        order = sorted(
            range(len(scores)),
            key=lambda item: (
                -float(scores[item]),
                tuple(float(value) for value in poses[item].reshape(-1)),
            ),
        )
        return tuple(
            GraspPoseCandidate(
                confidence=float(scores[item]),
                camera_from_gripper=tuple(
                    tuple(float(value) for value in row) for row in poses[item]
                ),
            )
            for item in order
            if float(scores[item]) >= minimum_confidence
        )[:maximum_count]
    return ()


__all__ = [
    "GraspGenXClient",
    "GraspGenXSettings",
    "GraspPoseCandidate",
    "GRASPGENX_REPRODUCIBLE_DECIMALS",
    "NoGraspPoseError",
    "UNITREE_G1_PREGRASP_STANDOFF_M",
    "UNITREE_G1_SWEEP_VOLUME",
    "UNITREE_G1_GRIPPER_FROM_DEX3_PALM",
    "grasp_pose_from_payload",
    "grasp_pose_payload",
]
