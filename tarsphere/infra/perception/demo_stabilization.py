"""Bounded offline ego stabilization; never a sensor or camera-pose correction.

Estimate at 640x480, apply each exact affine to original native pixels, then
compose HUD/map/text. The constant crop is separate from fail-closed identity
motion correction. This is image-space presentation, not measured 3D geometry.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StabilizationSettings:
    native_size: tuple = (2560, 1920)
    working_size: tuple = (640, 480)
    crop_per_edge: float = 0.03
    max_shift_px: float = 20.0
    max_roll_deg: float = 0.6
    smooth_sigma_frames: float = 2.0
    min_inliers: int = 35
    min_inlier_ratio: float = 0.65
    max_step_shift_px: float = 48.0
    max_step_roll_deg: float = 1.5
    preset: str = "light"

    @classmethod
    def for_preset(cls, preset="light", *, crop_per_edge=None):
        """Presentation presets; legacy light remains the default."""
        if preset == "natural":
            # Small, partial correction of only very short camera jitter.
            return cls(
                preset="natural",
                crop_per_edge=0.01 if crop_per_edge is None else crop_per_edge,
                max_shift_px=12.0,
                max_roll_deg=0.2,
                smooth_sigma_frames=1.0,
            ).validate()
        if preset == "light":
            return cls(
                **({} if crop_per_edge is None else {"crop_per_edge": crop_per_edge})
            ).validate()
        if preset == "gimbal":
            return cls(
                preset="gimbal",
                crop_per_edge=0.065 if crop_per_edge is None else crop_per_edge,
                max_shift_px=80.0,
                max_roll_deg=2.0,
                smooth_sigma_frames=9.0,
                min_inliers=40,
                min_inlier_ratio=0.42,
                max_step_shift_px=110.0,
                max_step_roll_deg=3.0,
            ).validate()
        raise ValueError(f"Unknown stabilization preset: {preset}")

    def validate(self):
        if self.native_size != (2560, 1920) or self.working_size != (640, 480):
            raise ValueError("Require original2560x1920 and estimation640x480")
        if self.preset == "natural":
            if (
                self.crop_per_edge != 0.01
                or self.max_shift_px != 12.0
                or self.max_roll_deg != 0.2
                or self.smooth_sigma_frames != 1.0
            ):
                raise ValueError(
                    "Natural requires fixed1percent crop,12px/.2deg bounds and1-frame smoothing"
                )
            return self
        if self.preset == "gimbal":
            if (
                self.crop_per_edge != 0.065
                or self.max_shift_px != 80.0
                or self.max_roll_deg != 2.0
                or self.smooth_sigma_frames != 9.0
            ):
                raise ValueError(
                    "Gimbal requires fixed6.5percent crop,80px/2deg bounds and9-frame smoothing"
                )
            return self
        if self.preset != "light":
            raise ValueError("Unknown stabilization preset")
        if not 0.025 <= self.crop_per_edge <= 0.035:
            raise ValueError("Fixed crop must be2.5–3.5 percent per edge")
        if not 0 <= self.max_shift_px <= 20 or not 0 <= self.max_roll_deg <= 0.6:
            raise ValueError("Native correction must stay within20px/.6deg")
        if not 0.5 <= self.smooth_sigma_frames <= 3:
            raise ValueError("Only a short high-frequency smoothing window is allowed")
        return self


def project_native_point(native_xy, matrix_2x3):
    """Source-native point -> stabilized-native point; use also for ABC leaders."""
    matrix = np.asarray(matrix_2x3, dtype=float)
    point = np.asarray(native_xy, dtype=float)
    if (
        matrix.shape != (2, 3)
        or point.shape != (2,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(point).all()
    ):
        raise ValueError("Expected finite native XY and2x3 affine")
    return (matrix @ np.r_[point, 1.0]).tolist()


def native_affine(shift, roll_rad, settings=StabilizationSettings()):
    """Constant center crop plus bounded rigid2D correction, no perspective warp."""
    settings.validate()
    shift = np.asarray(shift, dtype=float)
    if (
        shift.shape != (2,)
        or not np.isfinite(shift).all()
        or not math.isfinite(roll_rad)
    ):
        raise ValueError("Nonfinite correction")
    length = float(np.linalg.norm(shift))
    if length > settings.max_shift_px:
        shift *= settings.max_shift_px / length
    roll_rad = float(
        np.clip(
            roll_rad,
            -math.radians(settings.max_roll_deg),
            math.radians(settings.max_roll_deg),
        )
    )
    zoom = 1 / (1 - 2 * settings.crop_per_edge)
    center = (np.asarray(settings.native_size, dtype=float) - 1) / 2
    cosine, sine = math.cos(roll_rad), math.sin(roll_rad)
    linear = zoom * np.array([[cosine, -sine], [sine, cosine]])
    matrix = np.c_[linear, center + shift - linear @ center]
    if not covers_native_canvas(matrix, settings.native_size):
        raise ValueError("Fixed crop does not cover the native canvas")
    return matrix


def covers_native_canvas(matrix, native_size):
    """Every output corner inverse-maps inside source; affine convexity covers all."""
    width, height = native_size
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]],
        dtype=float,
    )
    source = (np.linalg.inv(full) @ corners.T).T
    return bool(
        np.isfinite(source).all()
        and (source[:, 0] >= 0).all()
        and (source[:, 0] <= width - 1).all()
        and (source[:, 1] >= 0).all()
        and (source[:, 1] <= height - 1).all()
    )


def relative_motion(previous, current, settings=StabilizationSettings()):
    """LK+forward/backward+RANSAC; reject large turns/scale/parallax/poor spread."""
    import cv2

    width, height = settings.working_size
    if previous.shape != (height, width) or current.shape != (height, width):
        raise ValueError("Estimation frames must be original-downsampled640x480 gray")
    mask = np.zeros_like(previous)
    mask[12 : int(height * 0.8), 12:-12] = 255
    gimbal = settings.preset == "gimbal"
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=800 if gimbal else 400,
        qualityLevel=0.012 if gimbal else 0.02,
        minDistance=6 if gimbal else 9,
        mask=mask,
    )
    fail = lambda reason: {
        "valid": False,
        "reason": reason,
        "step_native": [0.0, 0.0, 0.0],
    }
    if points is None or len(points) < settings.min_inliers:
        return fail("low_texture_identity_correction_reset")
    options = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    forward, good, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, **options
    )
    if forward is None:
        return fail("LK_failed_identity_correction_reset")
    backward, back_good, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, None, **options
    )
    if backward is None:
        return fail("backward_LK_failed_identity_correction_reset")
    usable = good.ravel().astype(bool) & back_good.ravel().astype(bool)
    usable &= (
        np.linalg.norm(points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1) <= 1.0
    )
    a, b = points.reshape(-1, 2)[usable], forward.reshape(-1, 2)[usable]
    if len(a) < settings.min_inliers:
        return fail("insufficient_tracks_identity_correction_reset")
    matrix, inliers = cv2.estimateAffinePartial2D(
        a,
        b,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0 if gimbal else 1.5,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return fail("RANSAC_failed_identity_correction_reset")
    chosen = inliers.ravel().astype(bool)
    count = int(chosen.sum())
    ratio = count / len(a)
    if count < settings.min_inliers or ratio < settings.min_inlier_ratio:
        return fail("low_consensus_identity_correction_reset")
    spread = np.ptp(a[chosen], axis=0)
    if spread[0] < width * (0.35 if gimbal else 0.45) or spread[1] < height * (
        0.20 if gimbal else 0.25
    ):
        return fail("local_foreground_or_poor_spread_identity_correction_reset")
    predicted = a[chosen] @ matrix[:, :2].T + matrix[:, 2]
    residual = float(np.median(np.linalg.norm(predicted - b[chosen], axis=1)))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle = math.atan2(matrix[1, 0], matrix[0, 0])
    center = (np.array([width, height]) - 1) / 2
    displacement = (matrix[:, :2] @ center + matrix[:, 2] - center) * 4
    if residual > (1.6 if gimbal else 1.2) or abs(scale - 1) > (
        0.025 if gimbal else 0.007
    ):
        return fail("nonrigid_or_scale_motion_identity_correction_reset")
    if (
        np.linalg.norm(displacement) > settings.max_step_shift_px
        or abs(math.degrees(angle)) > settings.max_step_roll_deg
    ):
        return fail("large_turn_or_step_identity_correction_reset")
    return {
        "valid": True,
        "reason": "background_affine_consensus",
        "step_native": [float(displacement[0]), float(displacement[1]), angle],
        "inliers": count,
        "tracked": len(a),
        "inlier_ratio": ratio,
        "residual_working_px": residual,
        "scale": scale,
    }


def estimate_motion(gray_frames, settings=StabilizationSettings()):
    settings.validate()
    previous = None
    motions = []
    for current in gray_frames:
        if previous is None:
            motion = {
                "valid": False,
                "reason": "first_frame_identity_correction_reset",
                "step_native": [0.0, 0.0, 0.0],
            }
        else:
            motion = relative_motion(previous, current, settings)
        motions.append(motion)
        previous = current
    return motions


def correction_plan(motions, settings=StabilizationSettings(), *, source_steps=None):
    """Symmetric short Gaussian trajectory residual; linear movement is preserved."""
    from scipy.ndimage import gaussian_filter1d

    settings.validate()
    corrections = np.zeros((len(motions), 3))
    segments = []
    start = None
    increments = np.asarray([m["step_native"] for m in motions], dtype=float)
    valid = np.asarray([m["valid"] for m in motions], dtype=bool)
    bridged = set()
    if settings.preset == "gimbal":
        # Interpolate only short, consistent tracking gaps, never source pixels or timestamps.
        index = 1
        while index < len(motions) - 1:
            if valid[index]:
                index += 1
                continue
            end = index
            while end < len(motions) and not valid[end]:
                end += 1
            if (
                end - index <= 2
                and end < len(motions)
                and valid[index - 1]
                and valid[end]
                and np.linalg.norm(increments[index - 1, :2] - increments[end, :2]) < 40
                and abs(increments[index - 1, 2] - increments[end, 2]) < math.radians(1)
            ):
                for k in range(index, end):
                    increments[k] = (
                        increments[index - 1] * (end - k)
                        + increments[end] * (k - index + 1)
                    ) / (end - index + 1)
                    valid[k] = True
                    bridged.add(k)
            index = end
    for index in range(len(motions) + 1):
        good = index < len(motions) and valid[index]
        if good and start is None:
            start = index
        if not good and start is not None:
            segments.append((start, index))
            start = None
    for start, end in segments:
        if end - start < 8:
            continue
        trajectory = np.cumsum(increments[start:end], axis=0)
        smooth = gaussian_filter1d(
            trajectory, settings.smooth_sigma_frames, axis=0, mode="nearest"
        )
        delta = smooth - trajectory
        # Fade to identity at segment edges; failed tracking never accumulates drift.
        fade = np.minimum(
            1.0,
            np.minimum(np.arange(end - start), np.arange(end - start)[::-1])
            / (6.0 if settings.preset == "gimbal" else 4.0),
        )
        # Natural preserves most walking motion: remove only a quarter of the
        # one-frame residual, with the strict light tracking gates and no gaps.
        gain = 0.25 if settings.preset == "natural" else 1.0
        corrections[start:end] = delta * fade[:, None] * gain
    rows = []
    for index, (motion, correction) in enumerate(zip(motions, corrections)):
        matrix = native_affine(correction[:2], correction[2], settings)
        center = (np.array(settings.native_size) - 1) / 2
        shift = np.asarray(project_native_point(center, matrix)) - center
        angle = math.atan2(matrix[1, 0], matrix[0, 0])
        rows.append(
            {
                "frame_index": index,
                "motion_estimate": motion,
                "correction_native_shift_px": shift.tolist(),
                "correction_roll_deg": math.degrees(angle),
                "affine_source_native_to_stabilized_native": matrix.tolist(),
                "fixed_crop_per_edge": settings.crop_per_edge,
                "canvas_fully_source_supported": True,
                "identity_motion_correction": bool(
                    np.allclose(shift, 0) and abs(angle) < 1e-12
                ),
            }
        )
        if settings.preset == "gimbal":
            rows[-1]["short_motion_estimate_gap_interpolated"] = index in bridged
    if source_steps is not None:
        if len(source_steps) != len(rows):
            raise ValueError("Source clock/affine count mismatch")
        for index in range(1, len(rows)):
            if source_steps[index] < source_steps[index - 1]:
                raise ValueError("Source captures must be monotonic")
            if source_steps[index] == source_steps[index - 1]:
                for key in (
                    "correction_native_shift_px",
                    "correction_roll_deg",
                    "affine_source_native_to_stabilized_native",
                    "identity_motion_correction",
                ):
                    rows[index][key] = rows[index - 1][key]
                rows[index]["native_capture_hold_affine_frozen"] = True
    return rows


def warp_native_frame(frame, matrix_2x3, settings=StabilizationSettings()):
    """Integration hook: raw native BGR -> warp -> labels/HUD; no intermediate encode."""
    import cv2

    width, height = settings.native_size
    matrix = np.asarray(matrix_2x3, dtype=float)
    if frame.shape != (height, width, 3) or not covers_native_canvas(
        matrix, settings.native_size
    ):
        raise ValueError("Require native pixels and a fully supported affine")
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def self_test():
    """Real LK synthetic recovery, smoothing, bounded canvas and reset checks."""
    import cv2

    settings = StabilizationSettings()
    rng = np.random.default_rng(17)
    texture = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.6)
    moved = cv2.warpAffine(texture, np.float64([[1, 0, 2], [0, 1, -1]]), (640, 480))
    recovered = relative_motion(texture, moved, settings)
    assert recovered["valid"] and np.allclose(
        recovered["step_native"][:2], [8, -4], atol=0.5
    )
    rotated = cv2.warpAffine(
        texture, cv2.getRotationMatrix2D((319.5, 239.5), 0.35, 1), (640, 480)
    )
    recovered_roll = relative_motion(texture, rotated, settings)
    assert (
        recovered_roll["valid"]
        and abs(math.degrees(recovered_roll["step_native"][2]) + 0.35) < 0.05
    )
    large = cv2.warpAffine(texture, np.float64([[1, 0, 18], [0, 1, 0]]), (640, 480))
    assert not relative_motion(texture, large, settings)["valid"]
    motions = [{"valid": False, "step_native": [0.0, 0.0, 0.0], "reason": "first"}]
    truth = []
    for i in range(40):
        truth.append(np.array([i * 2.0 + (-1) ** i * 5.0, 0.0, (-1) ** i * 0.004]))
    motions += [
        {
            "valid": True,
            "step_native": (truth[i] - truth[i - 1]).tolist(),
            "reason": "synthetic",
        }
        for i in range(1, 40)
    ]
    rows = correction_plan(motions, settings)
    assert len(rows) == len(motions) == 40
    original = np.array(truth)
    corrected = original + np.array(
        [
            r["correction_native_shift_px"] + [math.radians(r["correction_roll_deg"])]
            for r in rows
        ]
    )
    assert (
        np.std(np.diff(corrected[8:-8, 0])) < np.std(np.diff(original[8:-8, 0])) * 0.5
    )
    assert (
        abs((corrected[-1, 0] - corrected[0, 0]) - (original[-1, 0] - original[0, 0]))
        < 1e-6
    )
    for shift in [[20, 20], [-20, -20], [20, -20], [-20, 20]]:
        for angle in [-0.6, 0.6]:
            matrix = native_affine(shift, math.radians(angle), settings)
            assert covers_native_canvas(matrix, settings.native_size)
            center = (np.array(settings.native_size) - 1) / 2
            assert (
                np.linalg.norm(
                    np.asarray(project_native_point(center, matrix)) - center
                )
                <= 20 + 1e-9
            )
    flat = np.zeros((480, 640), np.uint8)
    assert not relative_motion(flat, flat, settings)["valid"]
    reset = correction_plan(
        [{"valid": False, "step_native": [0, 0, 0], "reason": "flat"}] * 10, settings
    )
    assert all(r["identity_motion_correction"] for r in reset)
    matrix = native_affine([7, -3], 0.002, settings)
    point = [1500.0, 1000.0]
    assert np.allclose(project_native_point(point, matrix), matrix @ [*point, 1.0])
    uniform = np.full((1920, 2560, 3), 100, np.uint8)
    warped = warp_native_frame(
        uniform, native_affine([20, 0], math.radians(0.6), settings), settings
    )
    assert (
        warped.shape == uniform.shape and warped.min() == 100 and np.all(uniform == 100)
    )
    natural = StabilizationSettings.for_preset("natural")
    natural_rows = correction_plan(motions, natural)
    natural_corrected = original + np.array(
        [
            r["correction_native_shift_px"] + [math.radians(r["correction_roll_deg"])]
            for r in natural_rows
        ]
    )
    retained = np.std(np.diff(natural_corrected[8:-8, 0])) / np.std(
        np.diff(original[8:-8, 0])
    )
    assert 0.7 < retained < 1.0
    assert (
        abs(
            (natural_corrected[-1, 0] - natural_corrected[0, 0])
            - (original[-1, 0] - original[0, 0])
        )
        < 1e-6
    )
    assert not relative_motion(texture, large, natural)["valid"]
    for shift in [[12, 12], [-12, -12], [12, -12], [-12, 12]]:
        for angle in [-0.2, 0.2]:
            matrix = native_affine(shift, math.radians(angle), natural)
            singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
            assert covers_native_canvas(matrix, natural.native_size)
            assert np.allclose(singular, [1 / 0.98, 1 / 0.98])
    return {
        "tests": "passed",
        "checks": [
            "actual_LK_synthetic_translation",
            "actual_LK_synthetic_roll",
            "large_step_rejected",
            "high_frequency_reduced_linear_motion_preserved",
            "exact_count",
            "20px_roll_crop_canvas_bounds",
            "low_texture_identity_reset",
            "ABC_exact_affine_projection",
            "native_pixel_warp_no_black_edge_source_unchanged",
            "natural_retains_majority_of_high_frequency_motion_and_full_travel",
            "natural_large_step_rejected",
            "natural_rigid_constant_scale_supported_canvas",
        ],
    }
