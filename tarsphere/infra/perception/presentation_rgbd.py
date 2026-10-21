"""Causal RGB-D point-cloud presentation with a floating online occupancy layer.

This is a read-only recording path. Colored points come from calibrated measured
depth pixels; the occupancy plane comes from the actual online mapper. Neither
scene meshes nor evaluator state are used.
"""

from __future__ import annotations

from pathlib import Path
from weakref import WeakKeyDictionary
import math

import numpy as np

_last_capture = WeakKeyDictionary()


def project_rgbd(rgb, depth, K, world_T_cam, *, max_points=30000):
    """Back-project real sampled pixels using the mapper's optical convention."""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    transform = np.asarray(world_T_cam, dtype=np.float64)
    if (
        depth.ndim != 2
        or rgb.ndim != 3
        or rgb.shape[:2] != depth.shape
        or rgb.shape[2] < 3
    ):
        raise ValueError("RGB and depth must share one calibrated HxW capture")
    if (
        K.shape != (3, 3)
        or transform.shape != (4, 4)
        or not np.isfinite(K).all()
        or not np.isfinite(transform).all()
    ):
        raise ValueError("RGB-D presentation requires finite camera calibration")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or max_points <= 0:
        raise ValueError("Focal lengths and point budget must be positive")
    stride = max(1, math.ceil(math.sqrt(depth.size / max_points)))
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    rows = rows.ravel()
    cols = cols.ravel()
    d = depth[rows, cols]
    valid = np.isfinite(d) & (d >= 0.1) & (d <= 8.0)
    rows = rows[valid]
    cols = cols[valid]
    d = d[valid].astype(np.float64)
    camera = np.c_[
        d * (cols - K[0, 2]) / K[0, 0],
        d * (rows - K[1, 2]) / K[1, 1],
        d,
        np.ones(len(d)),
    ]
    xyz = (camera @ transform.T)[:, :3].astype(np.float32)
    return (
        xyz,
        np.asarray(rgb[rows, cols, :3], dtype=np.uint8),
        np.c_[cols, rows].astype(np.int32),
    )


def capture_rgbd_snapshot(
    grid, root, physics_step, *, rgb, depth, K, world_T_cam, robot_xy, robot_yaw_rad
):
    """Save one synchronized point sample and online map at most twice per second."""
    if grid.map_source != "online":
        return None
    root = Path(root) / "presentation" / "rgbd"
    stamp = (str(root), int(physics_step) // 120)
    if _last_capture.get(grid) == stamp and root.exists():
        return None
    xyz, colors, pixels = project_rgbd(rgb, depth, K, world_T_cam)
    # Use the mapper's body footprint to remove robot self-returns.
    # This only filters displayed sensor points; it never changes the mapper.
    from tarsphere.infra.perception.online_mapping import (
        _G1_SELF_FILTER_HULL_XY,
        _points_in_padded_polygon,
    )
    from tarsphere.infra.control.physical_contracts import G1_RGBD_SELF_FILTER_PADDING_M

    robot_xy = np.asarray(robot_xy, dtype=np.float64)[:2]
    delta = xyz[:, :2] - robot_xy
    c, s = math.cos(robot_yaw_rad), math.sin(robot_yaw_rad)
    body = np.c_[c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]]
    self_return = _points_in_padded_polygon(
        body, _G1_SELF_FILTER_HULL_XY, padding_m=G1_RGBD_SELF_FILTER_PADDING_M
    )
    # Reference plane is the mapper's lower obstacle band, not an estimated floor.
    floor_z = float(grid.obstacle_z_min_m)
    keep = (~self_return) & (xyz[:, 2] >= floor_z - 0.15) & (xyz[:, 2] <= floor_z + 2.1)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{int(physics_step):012d}.npz"
    np.savez_compressed(
        path,
        physics_step=int(physics_step),
        map_source="online",
        points_xyz=xyz[keep],
        points_rgb=colors[keep],
        sampled_pixels_xy=pixels[keep],
        K=np.asarray(K),
        world_T_cam=np.asarray(world_T_cam),
        source_resolution=np.asarray(depth.shape[::-1]),
        grid=grid._grid,
        observed=grid._observed,
        origin_xy=grid._origin,
        resolution_m=grid.resolution,
        floor_z_m=floor_z,
        robot_xy=robot_xy,
        robot_yaw_rad=float(robot_yaw_rad),
        pose_history_xy=np.asarray(grid.exploration_pose_history()).reshape(-1, 2),
    )
    _last_capture[grid] = stamp
    return path


class LayeredOnlineMap:
    """Incremental observed RGB point history and the latest sensor-derived map."""

    def __init__(self):
        self.xyz = np.empty((0, 3), np.float32)
        self.rgb = np.empty((0, 3), np.uint8)
        self.voxel_m = 0.04
        self.state = None
        self.bounds = None
        self.last_step = -1

    def integrate(self, path):
        with np.load(path, allow_pickle=False) as data:
            state = {key: data[key].copy() for key in data.files}
        if str(state["map_source"].item()) != "online":
            raise ValueError("Layered presentation rejects ground-truth occupancy")
        step = int(state["physics_step"])
        if Path(path).stem.isdigit() and int(Path(path).stem) != step:
            raise ValueError("RGB-D filename and recorded physics step disagree")
        if step < self.last_step:
            raise ValueError("RGB-D snapshots must be chronological")
        xyz = np.asarray(state["points_xyz"], dtype=np.float32).reshape(-1, 3)
        rgb = np.asarray(state["points_rgb"], dtype=np.uint8).reshape(-1, 3)
        if len(xyz) != len(rgb) or not np.isfinite(xyz).all():
            raise ValueError("Invalid colored point capture")
        self.xyz = np.concatenate((self.xyz, xyz))
        self.rgb = np.concatenate((self.rgb, rgb))
        # A spatial voxel keeps the latest actual point/color, never synthesized RGB.
        while len(self.xyz):
            keys = np.floor(self.xyz / self.voxel_m).astype(np.int64)
            _, reverse_indices = np.unique(keys[::-1], axis=0, return_index=True)
            indices = len(self.xyz) - 1 - reverse_indices
            self.xyz = self.xyz[indices]
            self.rgb = self.rgb[indices]
            if len(self.xyz) <= 180000:
                break
            self.voxel_m *= 1.25
        robot = np.asarray(state["robot_xy"])
        if self.bounds is None:
            self.bounds = np.array([robot - 1.5, robot + 1.5], dtype=np.float64)
        if len(xyz):
            self.bounds[0] = np.minimum(self.bounds[0], xyz[:, :2].min(axis=0) - 0.3)
            self.bounds[1] = np.maximum(self.bounds[1], xyz[:, :2].max(axis=0) + 0.3)
        # Include only the CURRENT observed map extent, not a future scene extent.
        rows, cols = np.nonzero(state["observed"])
        if len(rows):
            origin = state["origin_xy"]
            res = float(state["resolution_m"])
            low = origin + np.array([rows.min(), cols.min()]) * res - 0.2
            high = origin + np.array([rows.max(), cols.max()]) * res + 0.2
            self.bounds[0] = np.minimum(self.bounds[0], low)
            self.bounds[1] = np.maximum(self.bounds[1], high)
        self.state = state
        self.last_step = step

    def render(self, resolution):
        from PIL import Image, ImageDraw, ImageFont

        w, h = resolution
        q = w / 288
        canvas = Image.new("RGBA", (w, h), "#1C1E27")
        draw = ImageDraw.Draw(canvas)

        def font(size, bold=False):
            for p in [
                (
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                    if bold
                    else "/System/Library/Fonts/Supplemental/Arial.ttf"
                ),
                (
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                    if bold
                    else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ),
            ]:
                if Path(p).is_file():
                    return ImageFont.truetype(p, max(8, round(size * q)))
            return ImageFont.load_default(size=max(8, round(size * q)))

        draw.rounded_rectangle(
            (0, 0, w - 1, h - 1),
            radius=12 * q,
            outline="#383442",
            width=max(1, round(q * 0.6)),
        )
        draw.text(
            (14 * q, 10 * q),
            "SEMANTIC SPATIAL MEMORY",
            font=font(10, True),
            fill="#E9E5EF",
        )
        draw.text((14 * q, 26 * q), "RGB-D + 2D", font=font(8), fill="#A6A0B0")
        if self.state is None:
            draw.text(
                (14 * q, h * 0.5),
                "Awaiting RGB-D capture",
                font=font(10),
                fill="#B9B4C0",
            )
            return canvas.convert("RGB")
        s = self.state
        low, high = self.bounds
        center = (low + high) / 2
        scale = (w - 30 * q) / (0.7071 * np.sum(high - low))
        # The occupancy sheet sits just above the captured room-height band,
        # rather than occupying a separate chart row far above the cloud.
        floor = float(s["floor_z_m"])
        layer_gap = min(h * 0.24, 2.45 * scale * 0.85)
        mid_y = (38 * q + h - 29 * q) / 2
        top_y = mid_y - layer_gap / 2
        cloud_y = mid_y + layer_gap / 2
        matrix = np.array([[0.7071, -0.7071], [0.19, 0.19]]) * scale

        def xy(points, y):
            return (np.asarray(points) - center) @ matrix.T + np.array([w * 0.5, y])

        corners = np.array(
            [[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]
        )
        # Sparse reference lattice under the measured cloud, no solid platform.
        grid_layer = Image.new("RGBA", (w, h))
        g = ImageDraw.Draw(grid_layer)
        for xx in np.linspace(low[0], high[0], 7):
            g.line(
                [tuple(p) for p in xy([[xx, low[1]], [xx, high[1]]], cloud_y)],
                fill=(169, 157, 193, 30),
                width=max(1, round(q * 0.45)),
            )
        for yy in np.linspace(low[1], high[1], 7):
            g.line(
                [tuple(p) for p in xy([[low[0], yy], [high[0], yy]], cloud_y)],
                fill=(169, 157, 193, 30),
                width=max(1, round(q * 0.45)),
            )
        canvas = Image.alpha_composite(canvas, grid_layer)
        # A z-buffer in display space retains a visible sampled RGB point per pixel.
        if len(self.xyz):
            points = xy(self.xyz[:, :2], cloud_y)
            points[:, 1] -= (self.xyz[:, 2] - floor) * scale * 0.85
            points = np.rint(points).astype(int)
            valid = (
                (points[:, 0] >= 0)
                & (points[:, 0] < w)
                & (points[:, 1] >= 30 * q)
                & (points[:, 1] < h - 18 * q)
            )
            points = points[valid]
            colors = self.rgb[valid]
            world = self.xyz[valid]
            # Farther XY first; near surfaces overwrite them. No semantic coloring.
            order = np.argsort(
                world[:, 0] + world[:, 1] + world[:, 2] * (2 * 0.19 / 0.85),
                kind="stable",
            )
            points = points[order]
            colors = colors[order]
            flat = points[:, 1] * w + points[:, 0]
            _, last = np.unique(flat[::-1], return_index=True)
            indices = len(flat) - 1 - last
            points = points[indices]
            colors = colors[indices]
            layer = np.zeros((h, w, 4), dtype=np.uint8)
            # Scale each point's covered area with output resolution; isolated
            # offset pixels would make the 4K cloud four times more transparent.
            radius = max(0, round(q * 0.28))
            offsets = [
                (dx, dy)
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
                if dx * dx + dy * dy <= radius * radius
            ]
            for dx, dy in offsets:
                xx = np.clip(points[:, 0] + dx, 0, w - 1)
                yy = np.clip(points[:, 1] + dy, 0, h - 1)
                layer[yy, xx, :3] = colors
                layer[yy, xx, 3] = 238
            canvas = Image.alpha_composite(canvas, Image.fromarray(layer))
        # Lift the exact online occupancy raster above the cloud with the SAME XY transform.
        occupancy = np.asarray(s["grid"])
        observed = np.asarray(s["observed"], dtype=bool)
        rgba = np.zeros((*occupancy.shape, 4), np.uint8)
        rgba[observed & (occupancy == -1)] = [156, 214, 188, 50]
        occupied = observed & (occupancy == 1)
        rgba[occupied] = [187, 153, 211, 76]
        padded = np.pad(occupied, 1)
        interior = (
            padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
        rgba[occupied & ~interior] = [209, 178, 234, 182]
        rgba[observed & (occupancy == 0)] = [154, 151, 170, 30]
        texture = Image.fromarray(rgba.transpose(1, 0, 2))
        inverse = np.linalg.inv(matrix) / float(s["resolution_m"])
        # Mapper indices denote cell centres; image coordinates denote pixel edges.
        shift = (
            (center - s["origin_xy"]) / float(s["resolution_m"])
            + 0.5
            - inverse @ np.array([w * 0.5, top_y])
        )
        coeff = (*inverse[0], shift[0], *inverse[1], shift[1])
        plane = texture.transform(
            (w, h), Image.Transform.AFFINE, coeff, resample=Image.Resampling.NEAREST
        )
        canvas = Image.alpha_composite(canvas, plane)
        guides = Image.new("RGBA", (w, h))
        g = ImageDraw.Draw(guides)
        for yy, alpha in [(top_y, 55), (cloud_y, 30)]:
            polygon = [tuple(p) for p in xy(corners, yy)]
            g.line(
                polygon + [polygon[0]],
                fill=(185, 170, 206, alpha),
                width=max(1, round(q * 0.6)),
            )
        trail = np.asarray(s["pose_history_xy"]).reshape(-1, 2)
        if len(trail) > 1:
            g.line(
                [tuple(p) for p in xy(trail, top_y)],
                fill=(202, 175, 216, 225),
                width=max(1, round(q)),
            )
        robot = np.asarray(s["robot_xy"]).reshape(1, 2)
        a = xy(robot, top_y)[0]
        b = xy(robot, cloud_y)[0]
        for fraction in np.arange(0, 1, 0.12):
            p0 = a + (b - a) * fraction
            p1 = a + (b - a) * min(1, fraction + 0.055)
            g.line(
                [tuple(p0), tuple(p1)],
                fill=(156, 214, 188, 100),
                width=max(1, round(q * 0.6)),
            )
        for p in [a, b]:
            r = 2.5 * q
            g.ellipse(
                (p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(156, 214, 188, 255)
            )
        canvas = Image.alpha_composite(canvas, guides)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (14 * q, h - 29 * q), "2D OCCUPANCY", font=font(8, True), fill="#B8A2CB"
        )
        draw.text(
            (144 * q, h - 29 * q), "RGB POINT CLOUD", font=font(8, True), fill="#9CD6BC"
        )
        draw.text((14 * q, h - 15 * q), "OBSERVED ONLINE", font=font(8), fill="#B9B4C0")
        return canvas.convert("RGB")


def write_layered_map_video(
    *, output_path, snapshot_dir, physics_steps, observer_fps, resolution
):
    from tarsphere.infra.perception.continuous_ego_video import _RawRgbVideoWriter

    physics_steps = tuple(int(step) for step in physics_steps)
    if any(b < a for a, b in zip(physics_steps, physics_steps[1:])):
        raise ValueError("Video physics steps must be chronological")
    paths = sorted(Path(snapshot_dir).glob("*.npz"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError("No recorded RGB-D map snapshots")
    writer = _RawRgbVideoWriter(
        output_path=output_path, resolution=resolution, fps=observer_fps
    )
    scene = LayeredOnlineMap()
    cursor = 0
    pixels = None
    try:
        for step in physics_steps:
            changed = False
            while cursor < len(paths) and int(paths[cursor].stem) <= int(step):
                scene.integrate(paths[cursor])
                cursor += 1
                changed = True
            if changed or pixels is None:
                pixels = np.asarray(scene.render(resolution))
            writer.write(pixels)
    finally:
        writer.close()
    return Path(output_path)
