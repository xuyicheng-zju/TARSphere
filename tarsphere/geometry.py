"""Camera projection and visibility geometry shared by grounding and evaluation.

Inputs are metric depth, calibrated camera intrinsics and pose, and evaluator
object volumes. A predicted image box selects valid depth pixels, which are
back-projected into the camera frame and transformed into world coordinates.
Hit rate and median-point distance measure localization; frustum tests establish
visibility. Agent perception backends do not supply evaluator geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


_EPS = 1e-9

# Shared evaluator/runtime dimensions for the AnimPeople collision proxy.
# Human roots are at the feet; the proxy extends upward from root z.
HUMAN_BODY_RADIUS_M = 0.20
HUMAN_BODY_HEIGHT_M = 1.70


# ---------------------------------------------------------------------------
# Camera intrinsics (spec §6.3: "camera intrinsics")
# ---------------------------------------------------------------------------


def fov_to_intrinsics(
    fov_deg: float,
    width: int,
    height: int,
    *,
    aspect: str = "horizontal",
) -> np.ndarray:
    """Build a 3x3 intrinsics matrix from FOV and resolution.

    Isaac's robot cameras are configured by horizontal FOV
    (``TARSPHERE_ROBOT_CAMERA_FOV_DEG``, default 100 — see isaac_env.py). The focal
    length that reproduces that FOV at the given aperture is

        focal_length = aperture / (2 * tan(FOV/2))

    (mirrors ``_set_camera_fov`` in isaac_env.py). We expose it as the
    pinhole intrinsics entries fx, fy with the principal point at the image
    centre.

    ``aspect`` selects how the vertical focal length is derived:
    - ``'horizontal'`` (default): fy = fx, treating pixels as square. This is
      the Isaac convention where only one FOV is authoritatively set.
    - ``'vertical'``: fy is derived from a vertical FOV equal to ``fov_deg``
      and fx = fy. Rarely what Isaac produces; kept for completeness.

    Returns a float64 3x3 matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if not (0.0 < fov_deg < 180.0):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")

    fov_rad = math.radians(fov_deg)
    cx = width / 2.0
    cy = height / 2.0

    if aspect == "horizontal":
        # aperture = width in pixels; fx reproduces the horizontal FOV.
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx
    elif aspect == "vertical":
        fy = (height / 2.0) / math.tan(fov_rad / 2.0)
        fx = fy
    else:
        raise ValueError(f"aspect must be 'horizontal' or 'vertical', got {aspect!r}")

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Camera pose (spec §6.3: "camera pose")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotBasePose:
    """Robot base pose as exposed by ``env.observe()['pose']``.

    ``position`` is the pelvis/world position [x, y, z]; ``yaw_rad`` is the
    heading. ``camera_height_m`` and ``camera_pitch_deg`` are the mounted
    camera's offset from the base (Isaac exposes both — see isaac_env.py
    pose payload).
    """

    position: tuple[float, float, float]
    yaw_rad: float
    camera_height_m: float = 1.0
    camera_pitch_deg: float = 0.0


def assemble_camera_pose(base: RobotBasePose) -> np.ndarray:
    """Assemble world_T_cam (4x4) from the robot base pose + camera mount.

    Convention: world axes are Isaac's (x-forward, y-left, z-up for the robot
    body frame after yaw rotation; camera optical frame is x-right, y-down,
    z-forward). The camera sits at the base xy translated up by
    ``camera_height_m`` and is pitched about the body-y axis by
    ``camera_pitch_deg``. Project convention (camera convention): a negative pitch
    looks down (H1WithHand uses -18 deg to view its manipulation area), so
    forward_z = sin(pitch) is negative for downward-looking mounts.

    The returned matrix maps homogeneous camera-frame points (x-right, y-down,
    z-forward) into world coordinates: ``p_world = world_T_cam @ p_cam``.
    """
    x, y, z = base.position
    # Camera optical centre starts directly above the base footprint at the
    # mount height. (z already encodes the pelvis/base height; the mount
    # height is the offset from there to the lens.)
    cam_origin = np.array([x, y, z + base.camera_height_m], dtype=np.float64)

    yaw = base.yaw_rad
    pitch = math.radians(base.camera_pitch_deg)

    # Body frame after yaw (about world z): x_forward rotates from +X by yaw.
    # cy, sy = cos(yaw), sin(yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Camera looks along body-x pitched about the body-y axis. The project's
    # convention (H1 camera: H1WithHand mount pitch -18 deg "looks down at
    # the manipulation area") is negative pitch = look down, so a negative
    # pitch must produce a downward (negative-z) forward component. Hence
    # forward_z = sin(pitch): pitch=-18 deg -> sin(-18 deg) < 0 -> down.
    forward = np.array(
        [
            cy * math.cos(pitch),
            sy * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    # Body +Y points left, so optical +X (image-right) is body -Y.
    cam_x = np.array([sy, -cy, 0.0], dtype=np.float64)
    # Camera y (down) = forward x cam_x (right-handed optical frame)
    cam_y = np.cross(forward, cam_x)

    R = np.stack([cam_x, cam_y, forward], axis=1)  # columns are the cam axes in world
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = cam_origin
    return T


def transform_to_world(points_cam: np.ndarray, world_T_cam: np.ndarray) -> np.ndarray:
    """Transform Nx3 camera-frame points to world frame.

    ``points_cam`` are in the optical frame (x-right, y-down, z-forward) so
    they are the direct output of ``backproject_bbox``.
    """
    if points_cam.size == 0:
        return points_cam.reshape(0, 3)
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    homog = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    world = homog @ world_T_cam.T
    return world[:, :3]


# ---------------------------------------------------------------------------
# Depth validity (mirrors isaac_env.prepare_depth_frame)
# ---------------------------------------------------------------------------


def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    """Boolean mask of usable depth pixels (finite and strictly positive).

    Mirrors ``prepare_depth_frame`` (isaac_env.py): invalid pixels are
    non-finite or <= 0 and are zeroed there. Here we just return the mask.
    """
    d = np.asarray(depth)
    return np.isfinite(d) & (d > 0)


# ---------------------------------------------------------------------------
# Back-projection (spec §6.3: central crop -> valid pixels -> K^-1 lift)
# ---------------------------------------------------------------------------


def central_crop(
    bbox_xyxy: Sequence[float],
    alpha: float = 0.5,
) -> tuple[int, int, int, int]:
    """Inner crop of a 2D bbox, keeping the central ``alpha`` fraction.

    spec §6.3: ``C(b, alpha)`` with default ``alpha = 0.5``. A central crop
    rejects bbox edges (which often bleed into the background) before
    sampling depth. Returns integer pixel bounds [x1, x2, y1, y2] clipped to
    a non-empty region.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) * alpha / 2.0
    half_h = (y2 - y1) * alpha / 2.0
    nx1 = int(math.floor(cx - half_w))
    nx2 = int(math.ceil(cx + half_w))
    ny1 = int(math.floor(cy - half_h))
    ny2 = int(math.ceil(cy + half_h))
    if nx2 <= nx1:
        nx2 = nx1 + 1
    if ny2 <= ny1:
        ny2 = ny1 + 1
    return nx1, nx2, ny1, ny2


def backproject_bbox(
    bbox_xyxy: Sequence[float],
    depth: np.ndarray,
    K: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    """Back-project the central crop of a bbox to camera-frame 3D points.

    Implements spec §6.3 ``P_i = { D(u) K^-1 [u_x, u_y, 1]^T : u in U_i }``
    where ``U_i`` is the set of valid depth pixels inside the central crop.
    Returns an Nx3 array in the camera optical frame (x-right, y-down,
    z-forward). Empty if no valid pixels fall inside the crop.
    """
    depth = np.asarray(depth)
    h, w = depth.shape[:2]
    x1, x2, y1, y2 = central_crop(bbox_xyxy, alpha)
    # Clip to image bounds.
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3), dtype=np.float64)

    sub = depth[y1:y2, x1:x2]
    mask = valid_depth_mask(sub)
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float64)

    vs, us = np.nonzero(mask)  # rows, cols inside the crop
    us = us + x1
    vs = vs + y1
    d = sub[mask]

    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    fx, fy = K_inv[0, 0], K_inv[1, 1]
    cx_inv, cy_inv = K_inv[0, 2], K_inv[1, 2]
    # Manual application is cheaper than building [u,v,1] stacks for large crops.
    xs = d * (us * fx + cx_inv)
    ys = d * (vs * fy + cy_inv)
    zs = d
    return np.stack([xs, ys, zs], axis=1).astype(np.float64)


def backproject_mask(
    foreground_mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Back-project valid foreground pixels to camera-frame 3D points.

    A detector box contains both the object and whatever lies behind it. That
    background dominates depth for small tabletop objects and can move the
    estimated world coordinate onto the support surface. An instance mask is
    therefore the required metric support for semantic object grounding.
    """
    depth_array = np.squeeze(np.asarray(depth))
    mask_array = np.squeeze(np.asarray(foreground_mask))
    if depth_array.ndim != 2 or mask_array.ndim != 2:
        raise ValueError("foreground mask and depth must each resolve to HxW")
    if mask_array.shape != depth_array.shape:
        raise ValueError(
            "foreground mask and depth must describe the same image; "
            f"got {mask_array.shape} and {depth_array.shape}"
        )

    supported = (mask_array > 0) & valid_depth_mask(depth_array)
    if not np.any(supported):
        return np.zeros((0, 3), dtype=np.float64)

    rows, columns = np.nonzero(supported)
    ranges = depth_array[supported].astype(np.float64, copy=False)
    inverse_intrinsics = np.linalg.inv(np.asarray(K, dtype=np.float64))
    pixels = np.stack(
        [columns, rows, np.ones_like(columns)],
        axis=1,
    ).astype(np.float64, copy=False)
    return ((pixels @ inverse_intrinsics.T) * ranges[:, None]).astype(
        np.float64,
        copy=False,
    )


@dataclass(frozen=True)
class ObservedVolumeEvidence:
    """Exact RGB-D pixel support for one evaluator-owned 3-D volume."""

    bbox_xyxy: tuple[float, float, float, float]
    supported_pixel_count: int


def observed_volume_evidence(
    volume: "BBox3D",
    depth: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    *,
    tolerance_m: float = 0.05,
    min_pixels: int = 4,
    min_bbox_width_px: int = 1,
    min_bbox_height_px: int = 1,
) -> ObservedVolumeEvidence | None:
    """Return visible-pixel evidence supported by depth inside a GT volume.

    This is an evaluator-side visibility primitive, not dynamic 2-D ground
    truth.  Every valid depth pixel is lifted through the calibrated camera;
    pixels whose world points fall inside the slightly expanded 3-D volume
    are the observable support of that volume in the current frame.  The
    resulting box therefore excludes geometry hidden behind walls, unlike a
    projection-only AABB or Replicator's occlusion-agnostic tight box.

    ``min_pixels`` rejects isolated depth noise while remaining resolution
    independent at the benchmark camera sizes.  Coordinates use the same
    half-open ``[x1, y1, x2, y2]`` convention consumed by
    :func:`backproject_bbox`.
    """
    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be non-negative.")
    if min_pixels <= 0:
        raise ValueError("min_pixels must be positive.")
    if min_bbox_width_px <= 0 or min_bbox_height_px <= 0:
        raise ValueError("minimum bbox dimensions must be positive.")

    depth_array = np.asarray(depth)
    if depth_array.ndim < 2:
        return None
    depth_array = np.squeeze(depth_array)
    if depth_array.ndim != 2:
        return None
    valid = valid_depth_mask(depth_array)
    if not np.any(valid):
        return None

    rows, columns = np.nonzero(valid)
    ranges = depth_array[valid].astype(np.float64, copy=False)
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    pixels = np.stack(
        [columns, rows, np.ones_like(columns)],
        axis=1,
    ).astype(np.float64, copy=False)
    points_camera = (pixels @ K_inv.T) * ranges[:, None]
    points_world = transform_to_world(points_camera, world_T_cam)
    supported = volume.expanded(float(tolerance_m)).contains(points_world)
    supported_pixel_count = int(np.count_nonzero(supported))
    if supported_pixel_count < int(min_pixels):
        return None

    visible_columns = columns[supported]
    visible_rows = rows[supported]
    bbox = (
        float(np.min(visible_columns)),
        float(np.min(visible_rows)),
        float(np.max(visible_columns) + 1),
        float(np.max(visible_rows) + 1),
    )
    if bbox[2] - bbox[0] < int(min_bbox_width_px) or bbox[3] - bbox[1] < int(
        min_bbox_height_px
    ):
        return None
    return ObservedVolumeEvidence(
        bbox_xyxy=bbox,
        supported_pixel_count=supported_pixel_count,
    )


def observed_volume_bbox(
    volume: "BBox3D",
    depth: np.ndarray,
    K: np.ndarray,
    world_T_cam: np.ndarray,
    *,
    tolerance_m: float = 0.05,
    min_pixels: int = 4,
) -> tuple[float, float, float, float] | None:
    """Compatibility-free convenience view of RGB-D volume evidence."""

    evidence = observed_volume_evidence(
        volume,
        depth,
        K,
        world_T_cam,
        tolerance_m=tolerance_m,
        min_pixels=min_pixels,
    )
    return None if evidence is None else evidence.bbox_xyxy


# ---------------------------------------------------------------------------
# GT volumes (spec §6.3: expanded tolerant volume V_i^{+delta})
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox3D:
    """Axis-aligned 3D bounding box from the static scene graph.

    ``min`` / ``max`` are inclusive corners. ``center`` and ``size`` are
    derived. The scene-graph builder emits this shape under each node's
    ``bbox`` payload (see scene_graph/builder.py ``_bbox_payload``).
    """

    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @property
    def center(self) -> np.ndarray:
        return (np.array(self.min) + np.array(self.max)) / 2.0

    @property
    def size(self) -> np.ndarray:
        return np.array(self.max) - np.array(self.min)

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.size))

    def expanded(self, delta: float) -> "BBox3D":
        """Return a copy grown by ``delta`` in every direction."""
        mn = tuple(c - delta for c in self.min)
        mx = tuple(c + delta for c in self.max)
        return BBox3D(mn, mx)  # type: ignore[arg-type]

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: which points fall inside this AABB."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return np.zeros(0, dtype=bool)
        mn = np.array(self.min)
        mx = np.array(self.max)
        return np.all((pts >= mn) & (pts <= mx), axis=1)

    def distance_to(self, points: np.ndarray) -> np.ndarray:
        """Per-point Euclidean distance to the box surface (0 if inside)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return np.zeros(0, dtype=np.float64)
        mn = np.array(self.min)
        mx = np.array(self.max)
        clamped = np.minimum(np.maximum(pts, mn), mx)
        return np.linalg.norm(pts - clamped, axis=1)


def human_bbox3d(position: Sequence[float]) -> BBox3D:
    """Construct the evaluator human AABB from one runtime root position."""
    if len(position) < 3:
        raise ValueError(f"human position must contain xyz, got {position!r}")
    x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    return BBox3D(
        min=(x - HUMAN_BODY_RADIUS_M, y - HUMAN_BODY_RADIUS_M, z),
        max=(x + HUMAN_BODY_RADIUS_M, y + HUMAN_BODY_RADIUS_M, z + HUMAN_BODY_HEIGHT_M),
    )


def tolerant_delta(volume: BBox3D) -> float:
    """spec §6.3: delta_i = max(0.05 m, 0.1 * diameter(V_i^*))."""
    return max(0.05, 0.1 * volume.diameter)


# ---------------------------------------------------------------------------
# 3D localisation scoring (spec §6.3: h_i, c_i, m_i^3D)
# ---------------------------------------------------------------------------


@dataclass
class LocalizationScore:
    """Sub-terms of ``bbox_to_3d_volume`` for one object.

    - ``hit_rate`` (h_i): fraction of lifted points inside V_i^{+delta}.
    - ``center_score`` (c_i): exp(-d_i^2 / (2 sigma^2)) where d_i is the
      distance from the median lifted point to V_i^{+delta} and sigma =
      delta_i.
    - ``m_3d`` (m_i^3D): sqrt(h_i * c_i). The caller gates this on GT
      visibility and predicted presence (spec §6.3 indicator terms).
    """

    hit_rate: float = 0.0
    center_score: float = 0.0
    m_3d: float = 0.0
    n_points: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hit_rate": self.hit_rate,
            "center_score": self.center_score,
            "m_3d": self.m_3d,
            "n_points": self.n_points,
        }


def score_object_3d_localization(
    points_world: np.ndarray,
    gt_volume: BBox3D,
    *,
    delta: Optional[float] = None,
) -> LocalizationScore:
    """Compute spec §6.3 h_i, c_i, sqrt(h_i c_i) for one object.

    ``points_world`` are the bbox's lifted points already transformed into
    the world frame. Returns a LocalizationScore with all sub-terms. With no
    lifted points the score is zero (spec notes large wrong boxes already
    fail via low hit rate; an empty box is the degenerate limit).
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return LocalizationScore(hit_rate=0.0, center_score=0.0, m_3d=0.0, n_points=0)

    d = delta if delta is not None else tolerant_delta(gt_volume)
    expanded = gt_volume.expanded(d)

    inside = expanded.contains(pts)
    hit_rate = float(inside.sum()) / (n + _EPS)

    # spec §6.3: distance from the median lifted point to V_i^{+delta}.
    # When the median already lies inside the (expanded) volume the distance
    # is 0 -> centre score 1.
    median_pt = np.median(pts, axis=0)
    median_dist = float(expanded.distance_to(median_pt.reshape(1, 3))[0])

    sigma = d
    center_score = float(math.exp(-(median_dist**2) / (2.0 * sigma * sigma + _EPS)))

    m_3d = float(math.sqrt(max(0.0, hit_rate * center_score)))
    return LocalizationScore(
        hit_rate=hit_rate,
        center_score=center_score,
        m_3d=m_3d,
        n_points=n,
    )


# ---------------------------------------------------------------------------
# GT visibility (spec §6.2 v_i^*)
# ---------------------------------------------------------------------------


def is_visible_in_frustum(
    obj_volume: BBox3D,
    world_T_cam: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    *,
    depth: Optional[np.ndarray] = None,
    near_m: float = 0.1,
    far_m: float = 20.0,
) -> bool:
    """Decide whether a GT object volume is visible to the robot camera.

    A volume is visible when at least one of its corners projects inside the
    image and lands within [near, far] in front of the camera. When a depth
    map is supplied we additionally require the corner ray to be unoccluded
    by nearer geometry (the corner's range is below the corresponding depth
    pixel plus a tolerance). Without depth we fall back to frustum inclusion
    only — spec §6.2 needs a visibility flag, and a frustum test is the
    conservative default that does not depend on agent-reported perception.

    Used to produce GT ``v_i^*`` for ``presence`` and to gate the 3D
    localisation score in ``bbox_to_3d_volume`` (spec §6.3 notes:
    "If object is not visible in GT, localization is not evaluated").
    """
    h, w = image_size
    cam_T_world = np.linalg.inv(world_T_cam)

    # Sample the 8 AABB corners; visible if any corner is in-frustum.
    mn = np.array(obj_volume.min)
    mx = np.array(obj_volume.max)
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )
    homog = np.concatenate([corners, np.ones((8, 1))], axis=1)
    cam_pts = (cam_T_world @ homog.T).T[:, :3]

    # In optical frame z is forward; drop anything behind/near the camera.
    z = cam_pts[:, 2]
    in_range = (z >= near_m) & (z <= far_m)
    if not np.any(in_range):
        return False

    # Project to pixels: [u, v] = K @ [x, y, z] / z.
    K_arr = np.asarray(K, dtype=np.float64)
    proj = cam_pts[in_range] @ K_arr.T
    proj = proj[:, :2] / cam_pts[in_range, 2:3]
    u, v = proj[:, 0], proj[:, 1]
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(in_image):
        return False

    if depth is None:
        return True

    # Occlusion test: at least one in-image corner must be in front of the
    # surface measured at its pixel (within a small tolerance).
    depth_arr = np.asarray(depth)
    tol = 0.05  # 5 cm tolerance for the corner-vs-surface comparison
    for (uu, vv), zz in zip(proj[in_image], cam_pts[in_range, 2][in_image]):
        ui, vi = int(round(uu)), int(round(vv))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        d_pix = depth_arr[vi, ui]
        if not np.isfinite(d_pix) or d_pix <= 0:
            continue
        if zz <= d_pix + tol:
            return True
    return False


__all__ = [
    "HUMAN_BODY_RADIUS_M",
    "HUMAN_BODY_HEIGHT_M",
    "fov_to_intrinsics",
    "RobotBasePose",
    "assemble_camera_pose",
    "transform_to_world",
    "valid_depth_mask",
    "central_crop",
    "backproject_bbox",
    "backproject_mask",
    "observed_volume_bbox",
    "observed_volume_evidence",
    "ObservedVolumeEvidence",
    "BBox3D",
    "human_bbox3d",
    "tolerant_delta",
    "LocalizationScore",
    "score_object_3d_localization",
    "is_visible_in_frustum",
]
