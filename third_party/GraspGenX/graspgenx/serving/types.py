# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level types shared by the GraspGenX ZMQ client and server.

This module is deliberately torch-free so that client-only installs (numpy +
pyzmq + msgpack) can import it. Heavy imports (gripper asset resolution) are
deferred into the methods that need them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

# Order of the flat (12,) conditioning vector — matches the layout the model
# consumes as ``sweep_volume_open_and_mid`` (see grasp_server.load_gripper_input).
FLAT_FIELD_ORDER = ("extents_open", "offset_open", "extents_mid", "offset_mid")

GRIPPER_TYPE_MAP = {"parallel_2f": 0, "revolute_2f": 1, "revolute_3f": 2}


@dataclass
class SweepVolumeParams:
    """Sweep-volume conditioning parameters for a gripper ("sweep volume v2").

    Two axis-aligned boxes in the gripper base frame (+Z = approach axis,
    +X = closing direction), enclosing the inner finger volume at the open and
    half-open states:

    Attributes:
        extents_open: (3,) box extents (meters) at the fully-open state.
        offset_open:  (3,) box center offset (meters) at the fully-open state.
        extents_mid:  (3,) box extents at the half-open state.
        offset_mid:   (3,) box center offset at the half-open state.
        gripper_type: 0 = parallel_2f, 1 = revolute_2f, 2 = revolute_3f.
                      Only consumed by non-default checkpoints.
        fingertip_depth: Optional gripper-base → fingertip distance along +Z.
            Only used by the GraspMoE OBB branch. When omitted it is derived
            from the top plane of the open sweep box
            (``offset_open.z + extents_open.z / 2``), which overshoots the
            true fingertip by ~1 cm across the shipped grippers — well within
            the OBB branch's approach-axis sweep.
    """

    extents_open: np.ndarray
    offset_open: np.ndarray
    extents_mid: np.ndarray
    offset_mid: np.ndarray
    gripper_type: int = 0
    fingertip_depth: Optional[float] = None

    def __post_init__(self) -> None:
        for name in FLAT_FIELD_ORDER:
            raw = getattr(self, name)
            v = np.asarray(raw, dtype=np.float32).reshape(-1)
            if v.shape != (3,):
                raise ValueError(
                    f"SweepVolumeParams.{name} must have 3 elements; "
                    f"got shape {np.asarray(raw).shape}"
                )
            if not np.all(np.isfinite(v)):
                raise ValueError(
                    f"SweepVolumeParams.{name} must be finite; got {v.tolist()}"
                )
            setattr(self, name, v)
        for name in ("extents_open", "extents_mid"):
            v = getattr(self, name)
            if not np.all(v > 0):
                raise ValueError(
                    f"SweepVolumeParams.{name} must be strictly positive; "
                    f"got {v.tolist()}"
                )
        self.gripper_type = int(self.gripper_type)
        if self.gripper_type not in (0, 1, 2):
            raise ValueError(
                f"gripper_type must be 0 (parallel_2f), 1 (revolute_2f) or "
                f"2 (revolute_3f); got {self.gripper_type}"
            )
        if self.fingertip_depth is not None:
            self.fingertip_depth = float(self.fingertip_depth)
            if not np.isfinite(self.fingertip_depth) or self.fingertip_depth <= 0:
                raise ValueError(
                    f"fingertip_depth must be a positive finite float; "
                    f"got {self.fingertip_depth}"
                )

    @property
    def resolved_fingertip_depth(self) -> float:
        """Explicit fingertip depth, or the open-box top plane as fallback."""
        if self.fingertip_depth is not None:
            return self.fingertip_depth
        return float(self.offset_open[2] + self.extents_open[2] / 2.0)

    @property
    def jaw_width(self) -> float:
        """Fingertip aperture (X extent of the open sweep box)."""
        return float(self.extents_open[0])

    def to_flat(self) -> np.ndarray:
        """(12,) float32 vector in FLAT_FIELD_ORDER — the model conditioning."""
        return np.concatenate(
            [getattr(self, name) for name in FLAT_FIELD_ORDER]
        ).astype(np.float32)

    @classmethod
    def from_flat(
        cls,
        flat,
        gripper_type: int = 0,
        fingertip_depth: Optional[float] = None,
    ) -> "SweepVolumeParams":
        v = np.asarray(flat, dtype=np.float32).reshape(-1)
        if v.shape != (12,):
            raise ValueError(
                f"Flat sweep-volume params must have 12 elements "
                f"({' + '.join(FLAT_FIELD_ORDER)}); got shape {np.asarray(flat).shape}"
            )
        return cls(
            extents_open=v[0:3],
            offset_open=v[3:6],
            extents_mid=v[6:9],
            offset_mid=v[9:12],
            gripper_type=gripper_type,
            fingertip_depth=fingertip_depth,
        )

    def to_dict(self) -> dict:
        """Wire representation (msgpack-friendly)."""
        d = {name: getattr(self, name) for name in FLAT_FIELD_ORDER}
        d["gripper_type"] = self.gripper_type
        if self.fingertip_depth is not None:
            d["fingertip_depth"] = self.fingertip_depth
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SweepVolumeParams":
        if not isinstance(d, dict):
            raise TypeError(f"Expected a dict; got {type(d).__name__}")
        allowed = set(FLAT_FIELD_ORDER) | {"gripper_type", "fingertip_depth"}
        unknown = set(d.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown sweep_volume_params keys: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}"
            )
        missing = set(FLAT_FIELD_ORDER) - set(d.keys())
        if missing:
            raise ValueError(f"sweep_volume_params missing keys: {sorted(missing)}")
        return cls(
            extents_open=d["extents_open"],
            offset_open=d["offset_open"],
            extents_mid=d["extents_mid"],
            offset_mid=d["offset_mid"],
            gripper_type=d.get("gripper_type", 0),
            fingertip_depth=d.get("fingertip_depth"),
        )

    @classmethod
    def coerce(
        cls, obj: Union["SweepVolumeParams", dict, np.ndarray, list, tuple]
    ) -> "SweepVolumeParams":
        """Accept a SweepVolumeParams, a wire dict, or a flat (12,) array."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        if isinstance(obj, (np.ndarray, list, tuple)):
            return cls.from_flat(obj)
        raise TypeError(
            f"sweep_volume_params must be a SweepVolumeParams, dict, or flat "
            f"(12,) array; got {type(obj).__name__}"
        )

    def cache_key(self) -> str:
        """Stable hash for server-side sampler caching."""
        flat = np.round(self.to_flat().astype(np.float64), 6)
        payload = (
            f"{flat.tolist()}|{self.gripper_type}|"
            f"{round(self.resolved_fingertip_depth, 6)}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_gripper_config(
        cls, gripper_name: str, assets_dir: Optional[str] = None
    ) -> "SweepVolumeParams":
        """Look up a named gripper's config.json and extract its params.

        Convenience for callers (and tests) that have the full graspgenx
        install: resolves the gripper asset directory the same way the
        name-based sampler does. Imports are deferred so client-only installs
        can still import this module.
        """
        import json
        import os

        from graspgenx.x_grippers import resolve_gripper_asset_dir

        asset_dir = resolve_gripper_asset_dir(gripper_name, assets_dir)
        with open(os.path.join(asset_dir, "config.json"), "r") as f:
            config = json.load(f)
        sv = config["sweep_volume"]
        return cls(
            extents_open=sv["extents"],
            offset_open=sv["offset"],
            extents_mid=sv["extents2"],
            offset_mid=sv["offset2"],
            gripper_type=GRIPPER_TYPE_MAP.get(config.get("type"), 0),
            fingertip_depth=config["fingertip"][-1],
        )
