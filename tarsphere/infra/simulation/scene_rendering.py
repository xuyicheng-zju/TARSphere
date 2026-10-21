"""Shared fixed rendering contract for evaluation and scene evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from tarsphere.infra.perception.sensor_conditions import (
    LOW_LIGHT_EXPOSURE_SCALE,
    PerceptualCondition,
)


EVALUATION_RENDER_SETTINGS: dict[str, Any] = {
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/histogram/whiteScale": 1.0,
    "/rtx/post/tonemap/filmIso": 80.0,
    "/rtx/post/tonemap/fNumber": 8.0,
    "/rtx/post/tonemap/exposureTime": 0.08,
    "/rtx/post/tonemap/cm2Factor": 1.0,
    # RTX enum 2 = Reinhard. Linear (1) clips bright white surfaces instead
    # of compressing their highlights; use the same curve for ego and BEV.
    "/rtx/post/tonemap/op": 2,
    "/rtx/ambientOcclusion/enabled": True,
    "/rtx/ambientOcclusion/intensity": 1.4,
    "/rtx/ambientOcclusion/radius": 3.0,
}


@dataclass(frozen=True)
class EvaluationSceneRendering:
    """Evidence that the complete evaluation rendering contract was applied."""

    render_settings: dict[str, Any]
    spatial_lighting_paths: tuple[str, ...]


def configure_evaluation_render_settings(
    settings: Any | None = None,
    *,
    perceptual_condition: PerceptualCondition | None = None,
) -> dict[str, Any]:
    """Apply the one fixed RTX exposure/lighting contract used in production."""

    if settings is None:
        import carb

        settings = carb.settings.get_settings()
    requested = dict(EVALUATION_RENDER_SETTINGS)
    if perceptual_condition == "low_light":
        requested["/rtx/post/tonemap/filmIso"] *= LOW_LIGHT_EXPOSURE_SCALE
    applied: dict[str, Any] = {}
    for key, value in requested.items():
        try:
            settings.set(key, value)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to set evaluation render setting {key}={value!r}: {exc}"
            ) from exc
        actual = settings.get(key)
        matches = (
            isinstance(actual, (int, float))
            and math.isclose(actual, value, rel_tol=1e-6)
            if isinstance(value, float)
            else actual == value
        )
        if not matches:
            raise RuntimeError(
                f"Evaluation render setting did not apply: {key} expected {value!r}, got {actual!r}"
            )
        applied[key] = value
    return applied


def apply_evaluation_scene_rendering(
    stage: Any,
    asset_path: str,
    *,
    perceptual_condition: PerceptualCondition | None = None,
    settings: Any | None = None,
) -> EvaluationSceneRendering:
    """Apply the atomic rendering contract shared by evaluation and BEV evidence."""

    from tarsphere.infra.simulation.scene_spatial_lighting import (
        apply_scene_spatial_lighting,
    )

    render_settings = configure_evaluation_render_settings(
        settings,
        perceptual_condition=perceptual_condition,
    )
    spatial_lighting_paths = apply_scene_spatial_lighting(stage, asset_path)
    return EvaluationSceneRendering(
        render_settings=render_settings,
        spatial_lighting_paths=spatial_lighting_paths,
    )


__all__ = [
    "EVALUATION_RENDER_SETTINGS",
    "EvaluationSceneRendering",
    "apply_evaluation_scene_rendering",
    "configure_evaluation_render_settings",
]
