#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for data preprocessing.
"""

import glob
import io
import json
import logging
import os
import time
from typing import Dict, Tuple

import h5py
import imageio
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import trimesh
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from graspgenx.dataset.eval_utils import (
    is_empty,
    load_h5_handle_empty_case,
    write_info,
    write_to_h5,
)
from graspgenx.dataset.exceptions import DataLoaderError
from graspgenx.dataset.dataset_utils import ObjectGraspDataset, load_grasp_data
from graspgenx.x_grippers import XGripperInfo

try:
    import cv2
except:
    pass


# ──────────────────────────────────────────────────────────────────────
# Flat-schema cache helpers
#
# On-disk grasps are stored as (N, 7) float32 [tx,ty,tz, qx,qy,qz,qw].
# Per-rendering visibility is stored as a bit-packed uint8 mask into the
# object's positive_grasps array, instead of replicating the grasps.
# Points are stored as float16 and upcast on load.
#
# In-memory shapes are unchanged: grasps stay (N, 4, 4) float32 and points
# stay (R, 3) float32, so every consumer in xgrasp_dataset.py continues to
# work without modification.
# ──────────────────────────────────────────────────────────────────────


def _grasps_to_7d(g):
    """Encode (N, 4, 4) SE(3) → (N, 7) float32 [tx,ty,tz, qx,qy,qz,qw].

    Empty / None inputs return a (0, 7) float32 array. Quaternion sign
    is canonicalized so qw >= 0 (handles the q/-q double cover).
    """
    if g is None:
        return np.empty((0, 7), dtype=np.float32)
    g = np.asarray(g)
    if g.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    assert g.ndim == 3 and g.shape[-2:] == (4, 4), f"expected (N,4,4), got {g.shape}"
    t = g[:, :3, 3]
    q_xyzw = Rotation.from_matrix(g[:, :3, :3]).as_quat()  # scipy returns xyzw
    # Canonical sign: qw >= 0
    flip = q_xyzw[:, 3] < 0
    if flip.any():
        q_xyzw = q_xyzw.copy()
        q_xyzw[flip] = -q_xyzw[flip]
    return np.concatenate([t, q_xyzw], axis=1).astype(np.float32, copy=False)


def _grasps_to_4x4(g):
    """Decode (N, 7) → (N, 4, 4) float32. Passthrough for (N, 4, 4) input.

    Empty / None inputs return a (0, 4, 4) float32 array.
    """
    if g is None:
        return np.empty((0, 4, 4), dtype=np.float32)
    g = np.asarray(g)
    if g.size == 0:
        return np.empty((0, 4, 4), dtype=np.float32)
    if g.ndim == 3 and g.shape[-2:] == (4, 4):
        return g.astype(np.float32, copy=False)
    assert g.ndim == 2 and g.shape[-1] == 7, f"expected (N,7) or (N,4,4), got {g.shape}"
    n = g.shape[0]
    out = np.zeros((n, 4, 4), dtype=np.float32)
    out[:, :3, 3] = g[:, :3]
    out[:, :3, :3] = Rotation.from_quat(g[:, 3:]).as_matrix()
    out[:, 3, 3] = 1.0
    return out


def _pack_rendering_mask(mask_list, n_pos):
    """Bit-pack a list of per-rendering bool masks into (R, ceil(N_pos/8)) uint8.

    Each mask is either None (interpreted as "all True" — mesh-mode
    rendering keeps every parent grasp) or a 1D bool array of length
    n_pos. Output is bit-packed along the grasp axis so storage per
    rendering is ~N_pos / 8 bytes.
    """
    r = len(mask_list)
    if r == 0 or n_pos == 0:
        return np.empty((r, 0), dtype=np.uint8)
    full = np.empty((r, n_pos), dtype=bool)
    for i, m in enumerate(mask_list):
        if m is None:
            full[i] = True
        else:
            m = np.asarray(m).astype(bool, copy=False)
            assert m.shape == (
                n_pos,
            ), f"rendering {i} mask shape {m.shape} != ({n_pos},)"
            full[i] = m
    return np.packbits(full, axis=1)


def _unpack_rendering_mask(packed, n_pos):
    """Unpack (R, ceil(N_pos/8)) uint8 → (R, N_pos) bool."""
    packed = np.asarray(packed)
    if packed.size == 0 or n_pos == 0:
        return np.empty((packed.shape[0], n_pos), dtype=bool)
    return np.unpackbits(packed, axis=1, count=n_pos).astype(bool)


def _key_to_h5(key: str) -> str:
    return key.replace("/", "____") if "/" in key else key


def _h5_to_key(key_h5: str) -> str:
    return key_h5.replace("____", "/") if "____" in key_h5 else key_h5


def _write_flat_object(
    grp: "h5py.Group",
    object_grasp_data: ObjectGraspDataset,
    rendering_output: list,
) -> None:
    """Write one object's data to the flat per-key H5 schema.

    Schema under each top-level key group:
        pg     (N_pos, 7)           float32   positive_grasps as 7D
        ng     (N_neg, 7)           float32   negative_grasps
        pg_op  (N_pos_op, 7)        float32   positive_grasps_onpolicy
        ng_op  (N_neg_op, 7)        float32   negative_grasps_onpolicy
        con    (N_con, 3)           float32   contacts
        pts    (R, P, 3)            float16   concatenated per-rendering points
        Tmv    (R, 4, 4)            float32   T_move_to_pc_mean per rendering
        rmask  (R, ceil(N_pos/8))   uint8     per-rendering visibility bitmask into pg
        rflag  (R, 3)               uint8     (mesh_mode, load_contact_batch, invalid) per rendering
      attrs:
        object_asset_path           str
        object_scale                float32
        R                           int       (number of renderings)
    """
    n_pos = (
        0
        if object_grasp_data.positive_grasps is None
        else len(object_grasp_data.positive_grasps)
    )

    pg = _grasps_to_7d(object_grasp_data.positive_grasps)
    ng = _grasps_to_7d(object_grasp_data.negative_grasps)
    pg_op = _grasps_to_7d(object_grasp_data.positive_grasps_onpolicy)
    ng_op = _grasps_to_7d(object_grasp_data.negative_grasps_onpolicy)

    con = object_grasp_data.contacts
    if con is None or len(con) == 0:
        con = np.empty((0, 3), dtype=np.float32)
    else:
        con = np.asarray(con, dtype=np.float32)

    r = len(rendering_output)
    if r == 0:
        pts = np.empty((0, 0, 3), dtype=np.float16)
        tmv = np.empty((0, 4, 4), dtype=np.float32)
        rmask = np.empty((0, 0), dtype=np.uint8)
        rflag = np.empty((0, 3), dtype=np.uint8)
    else:
        pts = np.stack(
            [np.asarray(rd["points"], dtype=np.float16) for rd in rendering_output],
            axis=0,
        )
        tmv = np.stack(
            [
                np.asarray(rd["T_move_to_pc_mean"], dtype=np.float32)
                for rd in rendering_output
            ],
            axis=0,
        )
        mask_list = [rd.get("grasp_visibility_mask") for rd in rendering_output]
        rmask = _pack_rendering_mask(mask_list, n_pos)
        rflag = np.stack(
            [
                np.array(
                    [
                        int(rd.get("mesh_mode", False)),
                        int(rd.get("load_contact_batch", False)),
                        int(rd.get("invalid", False)),
                    ],
                    dtype=np.uint8,
                )
                for rd in rendering_output
            ],
            axis=0,
        )

    grp.create_dataset("pg", data=pg)
    grp.create_dataset("ng", data=ng)
    grp.create_dataset("pg_op", data=pg_op)
    grp.create_dataset("ng_op", data=ng_op)
    grp.create_dataset("con", data=con)
    grp.create_dataset("pts", data=pts)
    grp.create_dataset("Tmv", data=tmv)
    grp.create_dataset("rmask", data=rmask)
    grp.create_dataset("rflag", data=rflag)

    grp.attrs["object_asset_path"] = str(object_grasp_data.object_asset_path)
    grp.attrs["object_scale"] = float(object_grasp_data.object_scale)
    grp.attrs["R"] = int(r)
    grp.attrs["n_pos"] = int(n_pos)


def _read_flat_object(grp: "h5py.Group") -> Tuple[ObjectGraspDataset, list]:
    """Read one object from the flat schema and reconstruct the in-memory
    (ObjectGraspDataset, list[rendering dict]) tuple that downstream code
    in xgrasp_dataset.py expects.

    Grasps come back as (N, 4, 4) float32. Points come back as float32.
    Per-rendering positive_grasps is reconstructed as parent[mask].
    """
    pg_7d = grp["pg"][...]
    n_pos = int(grp.attrs.get("n_pos", pg_7d.shape[0]))
    r = int(grp.attrs.get("R", 0))

    positive_grasps = _grasps_to_4x4(pg_7d)
    negative_grasps = _grasps_to_4x4(grp["ng"][...])
    positive_grasps_onpolicy = _grasps_to_4x4(grp["pg_op"][...])
    negative_grasps_onpolicy = _grasps_to_4x4(grp["ng_op"][...])
    contacts = np.asarray(grp["con"][...], dtype=np.float32)

    asset_path = grp.attrs["object_asset_path"]
    if isinstance(asset_path, bytes):
        asset_path = asset_path.decode("utf-8")
    object_scale = float(grp.attrs["object_scale"])

    object_grasp_data = ObjectGraspDataset(
        object_mesh=None,
        positive_grasps=positive_grasps,
        contacts=contacts,
        object_asset_path=asset_path,
        object_scale=object_scale,
        negative_grasps=negative_grasps,
        positive_grasps_onpolicy=positive_grasps_onpolicy,
        negative_grasps_onpolicy=negative_grasps_onpolicy,
    )

    renderings: list = []
    if r > 0:
        pts = grp["pts"][...].astype(np.float32, copy=False)
        tmv = grp["Tmv"][...].astype(np.float32, copy=False)
        rmask = grp["rmask"][...]
        rflag = grp["rflag"][...]
        masks = _unpack_rendering_mask(rmask, n_pos)
        for i in range(r):
            mask_i = masks[i]
            renderings.append(
                {
                    "mesh_mode": bool(rflag[i, 0]),
                    "load_contact_batch": bool(rflag[i, 1]),
                    "invalid": bool(rflag[i, 2]),
                    "points": pts[i],
                    "T_move_to_pc_mean": tmv[i],
                    "positive_grasps": positive_grasps[mask_i],
                }
            )

    return object_grasp_data, renderings


def _read_nested_object(grp: "h5py.Group") -> Tuple[ObjectGraspDataset, list]:
    """Read one object from the legacy nested schema (xgrasp_v1_largered_v2 and
    earlier). Each object group contains a ``grasp_data`` subgroup and a
    ``renderings`` subgroup of numbered rendering subgroups, with grasps stored
    as full (N, 4, 4) matrices (not 7-D compacted).
    """
    object_grasp_data = ObjectGraspDataset.from_dict(grp["grasp_data"])

    renderings: list = []
    for i in grp["renderings"].keys():
        rd = grp["renderings"][i]
        renderings.append(
            {
                "mesh_mode": rd["mesh_mode"][...].astype(np.bool_).item(),
                "load_contact_batch": rd["load_contact_batch"][...]
                .astype(np.bool_)
                .item(),
                "invalid": rd["invalid"][...].astype(np.bool_).item(),
                "points": rd["points"][...],
                "T_move_to_pc_mean": rd["T_move_to_pc_mean"][...],
                "positive_grasps": rd["positive_grasps"][...],
            }
        )

    return object_grasp_data, renderings


def _read_object(grp: "h5py.Group") -> Tuple[ObjectGraspDataset, list]:
    """Schema-detecting dispatcher: routes to the flat reader (xgrasp_v3+) or
    the nested reader (xgrasp_v1/v2) based on the keys present in the group.
    """
    if "pg" in grp:
        return _read_flat_object(grp)
    if "grasp_data" in grp:
        return _read_nested_object(grp)
    raise ValueError(
        f"Unknown H5 cache schema for object {grp.name}: "
        f"expected 'pg' (flat) or 'grasp_data' (nested), got {list(grp.keys())}"
    )


class XGraspJsonDatasetReader:
    """Class to efficiently read grasps data from JSON files in a regular directory structure."""

    def __init__(
        self,
        grasp_root_dir: str,
        object_root_dir: str,
        alternative_json_file_path: str = None,
    ):
        """
        Initialize the reader with grasp root directory and load/create the UUID to JSON path mapping.

        Args:
            grasp_root_dir (str): Path to directory containing grasp JSON files
            object_root_dir (str): Path to directory containing object files
            alternative_json_file_path (str, optional): If provided, load the
                UUID-to-path mapping from this path instead of the default
                location inside grasp_root_dir. Useful for reading from a
                local copy to avoid slow network filesystems.
        """
        self.grasp_root_dir = grasp_root_dir
        self.object_root_dir = object_root_dir
        self.map_uuid_to_path = {}

        if alternative_json_file_path and os.path.exists(alternative_json_file_path):
            mapping_file_path = alternative_json_file_path
        else:
            mapping_file_path = os.path.join(grasp_root_dir, "map_uuid_to_path.json")

        logger.info(
            f"Loading existing UUID to JSON path mapping from {mapping_file_path}"
        )
        try:
            with open(mapping_file_path, "r") as f:
                self.map_uuid_to_path = json.load(f)
        except Exception as e:
            print(
                f"[FATAL] Failed to load {mapping_file_path}: {type(e).__name__}: {e}",
                flush=True,
            )
            logger.error(
                f"[FATAL] Failed to load {mapping_file_path}: {type(e).__name__}: {e}"
            )
            raise
        logger.info(
            f"Loaded {len(self.map_uuid_to_path)} UUID to JSON path mappings from {mapping_file_path}."
        )

    def read_grasps_by_uuid(self, key_id: str) -> Union[Dict, None]:
        """
        Read grasps data for a specific object UUID.

        Args:
            key_id: {gripper}/{object}, the key to load the grasp data path

        Returns:
            Union[Dict, None]: Dictionary containing the grasps data if found, None otherwise
        """
        gripper_name = key_id.split("/")[0]
        object_id = key_id.split("/")[1]

        if (
            gripper_name not in self.map_uuid_to_path
            or object_id not in self.map_uuid_to_path[gripper_name]
        ):
            logger.debug(f"Object ID {object_id} not found in UUID mapping")
            return None

        json_file_basename = self.map_uuid_to_path[gripper_name][object_id]
        json_file_path = os.path.join(self.grasp_root_dir, json_file_basename)

        try:
            with open(json_file_path, "r") as f:
                grasps_dict = json.load(f)
            return grasps_dict
        except Exception as e:
            logger.error(f"Error loading grasps from {json_file_path}: {e}")
            return None


class XGraspGenDatasetCache:
    def __init__(self):
        self._cache = {}

    def __getitem__(self, key: str) -> Tuple[ObjectGraspDataset, dict]:
        return self._cache[key]

    def __setitem__(self, key: str, value: Tuple[ObjectGraspDataset, dict]):
        self._cache[key] = value

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get_keys(self):
        return self._cache.keys()

    def load_from_h5_file(self, path_to_h5_file: str):
        t0 = time.time()
        h5_file = h5py.File(path_to_h5_file, "r")

        for key_h5 in tqdm(
            h5_file.keys(), desc=f"Loading cache from H5 file: {path_to_h5_file}"
        ):
            self._cache[_h5_to_key(key_h5)] = _read_object(h5_file[key_h5])

        h5_file.close()
        logger.info(
            f"Loading cache from {path_to_h5_file} took {time.time() - t0}(s), with {len(self._cache)} keys"
        )

    def share_memory_(self):
        """Convert all numpy arrays in cache to shared-memory torch tensors.

        This enables efficient multi-process access in DDP training,
        avoiding per-process data duplication. All 8 GPU workers will
        read from the same physical memory (~250GB shared once).
        """
        for key in self._cache:
            object_grasp_data, renderings = self._cache[key]

            # Convert ObjectGraspDataset numpy arrays to shared tensors
            for attr_name in [
                "positive_grasps",
                "contacts",
                "negative_grasps",
                "positive_grasps_onpolicy",
                "negative_grasps_onpolicy",
            ]:
                val = getattr(object_grasp_data, attr_name, None)
                if val is not None and isinstance(val, np.ndarray):
                    setattr(
                        object_grasp_data,
                        attr_name,
                        torch.from_numpy(val).share_memory_(),
                    )

            # Convert rendering dict numpy arrays to shared tensors
            for rendering in renderings:
                for k, v in rendering.items():
                    if isinstance(v, np.ndarray):
                        rendering[k] = torch.from_numpy(v).share_memory_()

        return self

    def save_to_h5_file(self, path_to_h5_file: str):
        t0 = time.time()
        logger.info(f"Deleting old cache at {path_to_h5_file}")

        tmp_file = f"{path_to_h5_file[:-3]}_tmp.h5"
        if os.path.exists(tmp_file):
            os.system(f"rm {tmp_file}")  # For safety

        output_file = h5py.File(tmp_file, "a")
        for key, (object_grasp_data, rendering_output) in tqdm(
            self._cache.items(), desc=f"Saving cache to H5 file: {path_to_h5_file}"
        ):
            grp = output_file.create_group(_key_to_h5(key))
            _write_flat_object(grp, object_grasp_data, rendering_output)
        output_file.close()

        os.rename(tmp_file, path_to_h5_file)
        logger.info(f"Saving cache to {path_to_h5_file} took {time.time() - t0}(s)")


class XGraspGenDatasetCacheLazy:
    """Lazy HDF5-backed dataset cache — reads entries on demand.

    Instead of loading the entire (100–200 GB) H5 file into RAM, this
    class keeps only the list of keys in memory and reads individual
    entries from disk in __getitem__.

    Each DataLoader worker process lazily opens its own h5py.File handle
    (tracked via PID) to avoid cross-process file-handle issues after fork.
    """

    def __init__(self, path_to_h5_file: str):
        self._h5_path = path_to_h5_file
        self._h5_file = None  # opened lazily per-process
        self._pid = None  # detect fork / spawn

        # Try pre-computed key index first (milliseconds vs minutes on NFS)
        index_path = path_to_h5_file + ".keys.json"
        if os.path.exists(index_path):
            import json

            t0 = time.time()
            with open(index_path, "r") as f:
                h5_keys = json.load(f)
            self._keys = []
            self._h5_keys = {}
            for key_h5 in h5_keys:
                key = key_h5.replace("____", "/") if "____" in key_h5 else key_h5
                self._keys.append(key)
                self._h5_keys[key] = key_h5
            logger.info(
                f"Lazy cache: loaded {len(self._keys)} keys from index "
                f"{index_path} in {time.time() - t0:.2f}s"
            )
        else:
            # Fall back to scanning H5 keys (slow on NFS)
            t0 = time.time()
            with h5py.File(path_to_h5_file, "r") as f:
                self._keys = []
                self._h5_keys = {}
                for key_h5 in f.keys():
                    key = key_h5.replace("____", "/") if "____" in key_h5 else key_h5
                    self._keys.append(key)
                    self._h5_keys[key] = key_h5
            logger.info(
                f"Lazy cache: indexed {len(self._keys)} keys from "
                f"{path_to_h5_file} in {time.time() - t0:.1f}s "
                f"(no .keys.json found — consider running index_cache_h5.py)"
            )

    # ── per-process H5 handle ────────────────────────────────────────
    def _get_h5(self):
        """Get or create the per-process h5py File handle."""
        pid = os.getpid()
        if self._h5_file is None or self._pid != pid:
            # Close stale handle inherited across fork (if any)
            if self._h5_file is not None:
                try:
                    self._h5_file.close()
                except Exception:
                    pass
            self._h5_file = h5py.File(self._h5_path, "r")
            self._pid = pid
        return self._h5_file

    # ── pickle support (for spawn-based DataLoader workers) ──────────
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None  # h5py handles are not picklable
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ── dict-like interface (same as XGraspGenDatasetCache) ──────────
    def __getitem__(self, key: str) -> Tuple[ObjectGraspDataset, list]:
        h5 = self._get_h5()
        return _read_object(h5[self._h5_keys[key]])

    def __contains__(self, key: str) -> bool:
        return key in self._h5_keys

    def __len__(self):
        return len(self._keys)

    def get_keys(self):
        return self._keys

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


class ShardedH5Cache:
    """Progressive sharded HDF5 cache with background copying.

    Instead of copying a single monolithic H5 file (~500 GB) from NFS to local
    storage before training can start, this class works with per-gripper H5
    shard files (8–15 GB each).  A background thread copies shards from the
    NFS source directory to a local directory, and training can begin as soon
    as the first few shards are available.

    The class exposes the same dict-like interface as ``XGraspGenDatasetCache``
    and ``XGraspGenDatasetCacheLazy`` so it can be used as a drop-in
    replacement.

    All ranks on the same node see the same shards (copied to shared local
    storage).  Only ``local_rank == 0`` performs the actual file copies;
    other local ranks poll for newly-available shard files.  A standard
    ``DistributedSampler`` is used on top to partition samples across all
    DDP ranks (sample-level, not shard-level), ensuring balanced batch
    counts.

    Parameters
    ----------
    shard_dir_nfs : str
        Source directory on NFS containing per-gripper H5 shard files.
    shard_dir_local : str
        Local destination directory (e.g. ``/raid/scratch/xgrasp_cache/...``).
    shard_pattern : str
        Glob pattern relative to *shard_dir_nfs* that matches shard files.
        Example: ``"*__gripper_*.h5"``
    local_rank : int
        Local rank on this node (0–7).  Only local_rank 0 copies files;
        others poll for completion.
    min_shards_before_training : int
        Number of shards that must be available before
        ``wait_for_initial_shards`` returns.  Default 3.
    """

    def __init__(
        self,
        shard_dir_nfs: str,
        shard_dir_local: str,
        shard_pattern: str,
        local_rank: int = 0,
        min_shards_before_training: int = 3,
    ):
        import glob as _glob
        import threading

        self._shard_dir_nfs = shard_dir_nfs
        self._shard_dir_local = shard_dir_local
        self._local_rank = local_rank
        self._min_shards = min_shards_before_training

        # Discover all shard files on NFS
        all_shard_paths = sorted(_glob.glob(os.path.join(shard_dir_nfs, shard_pattern)))
        if not all_shard_paths:
            raise FileNotFoundError(
                f"No shard files matching '{shard_pattern}' in {shard_dir_nfs}"
            )
        self._all_shard_names = [os.path.basename(p) for p in all_shard_paths]
        logger.info(
            f"ShardedH5Cache: found {len(self._all_shard_names)} shards on NFS, "
            f"local_rank={local_rank}"
        )

        # State tracking
        self._lock = threading.Lock()
        self._available_shards: dict = {}  # name → XGraspGenDatasetCacheLazy
        self._available_keys: list = []  # flat list of keys from available shards
        self._key_to_shard: dict = {}  # key → shard name
        self._all_shards_ready = threading.Event()
        self._initial_shards_ready = threading.Event()
        self._copy_error = None

        os.makedirs(shard_dir_local, exist_ok=True)

        # Start background thread: copier (local_rank 0) or poller (others)
        if local_rank == 0:
            self._thread = threading.Thread(
                target=self._background_copy,
                daemon=True,
                name="shard-copier",
            )
        else:
            self._thread = threading.Thread(
                target=self._background_poll,
                daemon=True,
                name="shard-poller",
            )
        self._thread.start()

    # ── background copy (local_rank 0 only) ─────────────────────────
    def _background_copy(self):
        """Copy shards from NFS to local, registering each as it completes."""
        import shutil

        ready_count = 0
        total = len(self._all_shard_names)
        total_bytes_copied = 0

        for shard_name in self._all_shard_names:
            src = os.path.join(self._shard_dir_nfs, shard_name)
            dst = os.path.join(self._shard_dir_local, shard_name)
            done_marker = dst + ".done"

            try:
                src_size = os.path.getsize(src)

                # Skip copy if local file already exists with matching size
                # (no .done marker required — handles requeue to same node)
                if os.path.exists(dst):
                    dst_size = os.path.getsize(dst)
                    if dst_size == src_size:
                        logger.info(
                            f"ShardedH5Cache: shard {shard_name} already cached "
                            f"locally ({dst_size / 1e9:.1f} GB), skipping copy"
                        )
                        self._register_shard(shard_name, dst)
                        ready_count += 1
                        if ready_count >= min(self._min_shards, total):
                            self._initial_shards_ready.set()
                        continue

                t0 = time.time()
                tmp_dst = dst + ".tmp"
                shutil.copy2(src, tmp_dst)
                os.rename(tmp_dst, dst)
                # Write a done marker so other local ranks and future
                # requeueing can detect completed shards
                with open(done_marker, "w") as f:
                    f.write(str(src_size))
                elapsed = time.time() - t0
                total_bytes_copied += src_size

                logger.info(
                    f"ShardedH5Cache: copied shard {ready_count + 1}/{total} "
                    f"'{shard_name}' ({src_size / 1e9:.1f} GB) in {elapsed:.0f}s "
                    f"({total_bytes_copied / 1e9:.0f} GB total copied)"
                )

                self._register_shard(shard_name, dst)
                ready_count += 1

                if ready_count >= min(self._min_shards, total):
                    self._initial_shards_ready.set()

            except Exception as e:
                logger.error(
                    f"ShardedH5Cache: failed to copy shard '{shard_name}': {e}"
                )
                self._copy_error = e

        self._initial_shards_ready.set()
        self._all_shards_ready.set()
        logger.info(
            f"ShardedH5Cache: all {total} shards ready "
            f"({total_bytes_copied / 1e9:.0f} GB copied)"
        )

    # ── background poll (local_rank != 0) ───────────────────────────
    def _background_poll(self):
        """Poll for shards that local_rank 0 has finished copying."""
        total = len(self._all_shard_names)
        registered = set()

        while len(registered) < total:
            for shard_name in self._all_shard_names:
                if shard_name in registered:
                    continue
                dst = os.path.join(self._shard_dir_local, shard_name)
                done_marker = dst + ".done"
                if os.path.exists(dst) and (
                    os.path.exists(done_marker)
                    or os.path.getsize(dst)
                    == os.path.getsize(os.path.join(self._shard_dir_nfs, shard_name))
                ):
                    try:
                        self._register_shard(shard_name, dst)
                        registered.add(shard_name)
                        n = len(registered)
                        if n >= min(self._min_shards, total):
                            self._initial_shards_ready.set()
                    except Exception as e:
                        logger.warning(
                            f"ShardedH5Cache: poll failed for " f"'{shard_name}': {e}"
                        )
            if len(registered) < total:
                time.sleep(2)

        self._initial_shards_ready.set()
        self._all_shards_ready.set()
        logger.info(f"ShardedH5Cache: poller done — all {total} shards registered")

    def _register_shard(self, shard_name: str, local_path: str):
        """Index a newly-available shard and add its keys."""
        lazy = XGraspGenDatasetCacheLazy(local_path)
        new_keys = list(lazy.get_keys())

        with self._lock:
            if shard_name in self._available_shards:
                return  # already registered
            self._available_shards[shard_name] = lazy
            for key in new_keys:
                self._key_to_shard[key] = shard_name
            self._available_keys.extend(new_keys)

    # ── public API ──────────────────────────────────────────────────
    def wait_for_initial_shards(self, timeout: float = 600):
        """Block until min_shards_before_training shards are available."""
        logger.info(f"ShardedH5Cache: waiting for initial {self._min_shards} shards...")
        self._initial_shards_ready.wait(timeout=timeout)
        if self._copy_error is not None:
            logger.warning(f"ShardedH5Cache: copy error occurred: {self._copy_error}")
        with self._lock:
            n = len(self._available_shards)
            k = len(self._available_keys)
        logger.info(
            f"ShardedH5Cache: initial shards ready — " f"{n} shards, {k} keys available"
        )

    @property
    def all_shards_ready(self) -> bool:
        return self._all_shards_ready.is_set()

    def get_available_keys(self) -> list:
        """Return keys from shards that have been copied so far."""
        with self._lock:
            return list(self._available_keys)

    # ── dict-like interface (same as XGraspGenDatasetCache) ─────────
    def __getitem__(self, key: str):
        with self._lock:
            shard_name = self._key_to_shard[key]
            lazy = self._available_shards[shard_name]
        return lazy[key]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._key_to_shard

    def __len__(self):
        with self._lock:
            return len(self._available_keys)

    def get_keys(self):
        with self._lock:
            return list(self._available_keys)

    # ── pickle support (for DataLoader workers) ─────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_thread"] = None
        state["_initial_shards_ready"] = None
        state["_all_shards_ready"] = None
        return state

    def __setstate__(self, state):
        import threading

        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._initial_shards_ready = threading.Event()
        self._initial_shards_ready.set()
        self._all_shards_ready = threading.Event()
        if state.get("_copy_error") is None and len(self._available_shards) == len(
            self._all_shard_names
        ):
            self._all_shards_ready.set()

    @property
    def num_total_shards(self) -> int:
        return len(self._all_shard_names)

    @property
    def num_available_shards(self) -> int:
        with self._lock:
            return len(self._available_shards)


def load_object_xgrasp_data(
    key,
    object_root_dir,
    grasp_root_dir,
    min_grasps_gen=5,
    load_discriminator_dataset=False,
    onpolicy_dataset_json_dir=None,
    onpolicy_dataset_h5_path=None,
    onpolicy_json_path=None,
    onpolicy_data_found=False,
    grasp_dataset_reader: Union[XGraspJsonDatasetReader] = None,
) -> Tuple[DataLoaderError, Union[ObjectGraspDataset, None]]:

    if grasp_dataset_reader is not None:
        grasp_root_dir = None

    if onpolicy_dataset_json_dir is not None:
        if not onpolicy_data_found:
            onpolicy_json_path = None
            onpolicy_dataset_h5_path = None
            onpolicy_dataset_json_dir = None

    error_code, object_grasp_data = load_object_grasp_datapoint(
        key,
        object_root_dir,
        grasp_root_dir,
        load_discriminator_dataset=load_discriminator_dataset,
        onpolicy_dataset_json_dir=onpolicy_dataset_json_dir,
        onpolicy_dataset_h5_path=onpolicy_dataset_h5_path,
        onpolicy_json_path=onpolicy_json_path,
        grasp_dataset_reader=grasp_dataset_reader,
        min_pos_grasps_gen=min_grasps_gen,
    )
    return error_code, object_grasp_data


def load_object_grasp_datapoint(
    key_id: str,
    object_root_dir: str = None,
    grasp_root_dir: str = None,
    load_discriminator_dataset: bool = False,
    onpolicy_dataset_json_dir: str = None,
    onpolicy_dataset_h5_path: str = None,
    onpolicy_json_path: str = None,
    grasp_dataset_reader: Union[XGraspJsonDatasetReader] = None,
    min_pos_grasps_gen: int = 5,
) -> Tuple[DataLoaderError, Union[ObjectGraspDataset, None]]:
    """

    Args:
        object_root_dir: Root directory of the object dataset
        grasp_root_dir: Root directory of the grasp dataset
        object_id: Key of the object to load
        load_discriminator_dataset: Whether to load the discriminator dataset
        onpolicy_dataset_dir: Directory of the onpolicy dataset
        onpolicy_dataset_h5_path: Path to the onpolicy dataset h5 file
        onpolicy_json_path: Path to the onpolicy dataset json file
        grasp_dataset_reader: Dataset reader to use if the grasp dataset is stored in a webdataset or json format
        min_pos_grasps_gen: Minimum number of positive grasps for the generator dataset
    """

    positive_grasps_onpolicy = None
    negative_grasps_onpolicy = None
    onpolicy_object_scale = None
    has_onpolicy = (
        onpolicy_dataset_h5_path is not None and onpolicy_dataset_json_dir is not None
    )
    if has_onpolicy:
        positive_grasps_onpolicy, negative_grasps_onpolicy, onpolicy_object_scale = (
            load_onpolicy_dataset(
                key_id,
                onpolicy_dataset_h5_path,
                onpolicy_json_path,
            )
        )

    onpolicy_loaded = (
        positive_grasps_onpolicy is not None or negative_grasps_onpolicy is not None
    )

    error_code, grasps_dict = load_grasp_data(key_id, grasp_dataset_reader)

    offline_found = error_code == DataLoaderError.SUCCESS

    if not offline_found and not onpolicy_loaded:
        return DataLoaderError.GRASPS_FILE_NOT_FOUND_BOTH, None

    if not offline_found and onpolicy_loaded:
        logger.info(f"No offline grasps for {key_id}, using on-policy data only")

    if offline_found and has_onpolicy and not onpolicy_loaded:
        logger.info(f"No on-policy data for {key_id}, using offline grasps only")

    object_id = key_id.split("/")[1]
    uuid_object_paths_file = os.path.join(
        object_root_dir, "map_uuid_to_path_simplified.json"
    )

    uuid_to_path = json.load(open(uuid_object_paths_file))
    if object_id not in uuid_to_path:
        return DataLoaderError.UUID_NOT_FOUND_IN_MAPPING, None

    object_file = uuid_to_path[object_id]
    object_file = os.path.join(object_root_dir, object_file)
    if not os.path.exists(object_file):
        logger.error(f"Object mesh not found, at {object_file}")
        return DataLoaderError.OBJECT_MESH_NOT_FOUND, None

    if offline_found:
        object_scale = grasps_dict["object"]["scale"]
        grasps = grasps_dict["grasps"]
        grasp_poses = np.array(grasps["transforms"])
        grasp_mask = np.array(grasps["object_in_gripper"])
        positive_grasps = grasp_poses[grasp_mask]
        negative_grasps = grasp_poses[np.logical_not(grasp_mask)]
    else:
        if onpolicy_object_scale is None:
            logger.error(
                f"No object scale found for {key_id} (no offline grasps, "
                f"on-policy JSON missing object.scale)"
            )
            return DataLoaderError.OBJECT_SCALE_NOT_FOUND, None
        object_scale = onpolicy_object_scale
        positive_grasps = np.empty((0, 4, 4))
        negative_grasps = np.empty((0, 4, 4))

    contacts = None
    insufficient_grasps_warning = None
    if positive_grasps.shape[0] < min_pos_grasps_gen:
        logger.warning(
            f"Object {object_id} has too few offline grasps "
            f"(num pos:{len(positive_grasps)}, "
            f"neg:{len(negative_grasps) if negative_grasps is not None else 0}), "
            f"keeping datapoint anyway"
        )
        insufficient_grasps_warning = {
            "num_positive_grasps": int(len(positive_grasps)),
            "num_negative_grasps": (
                int(len(negative_grasps)) if negative_grasps is not None else 0
            ),
            "min_required": min_pos_grasps_gen,
        }

    try:
        object_mesh = trimesh.load(object_file)

        if type(object_mesh) == trimesh.Scene:
            object_mesh = object_mesh.dump(concatenate=True)

        object_mesh.apply_scale(object_scale)
    except:
        logger.debug(f"Unable to load object mesh at {object_file}")
        return DataLoaderError.OBJECT_MESH_LOAD_ERROR, None

    if contacts is None:
        cp = np.array([[0.0, 0, 0]])
        if len(positive_grasps) > 0:
            contacts = np.vstack([tra.transform_points(cp, g) for g in positive_grasps])

    if not offline_found:
        status = DataLoaderError.GRASPS_FILE_NOT_FOUND_OFFLINE
    elif has_onpolicy and not onpolicy_loaded:
        status = DataLoaderError.GRASPS_FILE_NOT_FOUND_ONLINE
    elif insufficient_grasps_warning:
        status = DataLoaderError.INSUFFICIENT_GRASPS_FOR_GENERATOR_DATASET
    else:
        status = DataLoaderError.SUCCESS

    obj_data = ObjectGraspDataset(
        object_mesh,
        positive_grasps,
        contacts,
        object_file,
        object_scale,
        negative_grasps=negative_grasps,
        positive_grasps_onpolicy=positive_grasps_onpolicy,
        negative_grasps_onpolicy=negative_grasps_onpolicy,
    )
    obj_data._insufficient_grasps_detail = insufficient_grasps_warning
    return status, obj_data


def filter_xgripper_grasps_by_point_cloud_visibility(
    grasps: np.ndarray,
    pointcloud: Union[np.ndarray, torch.Tensor],
    gripper_info: XGripperInfo,
) -> Union[np.ndarray, None]:
    """
    Grasps are assumed to be in the point cloud frame.

    Removes grasps are in the self-occluded regions (not visible from current camera pose) of the point cloud
    """
    num_grasps_initial = grasps.shape[0]

    if num_grasps_initial == 0:
        return None

    gripper_sweep_volume = torch.tensor(gripper_info.grasp_volume, dtype=torch.float32)
    grasps = torch.from_numpy(grasps).float()
    if type(pointcloud) == np.ndarray:
        pointcloud = torch.from_numpy(pointcloud).float()  # (N2, 3)

    homo_ptc = torch.cat([pointcloud, torch.ones_like(pointcloud[:, :1])], dim=-1)
    grasps_inv = torch.inverse(grasps)
    grasps_inv_ptc = grasps_inv @ (homo_ptc.T).unsqueeze(0)
    grasps_inv_ptc = grasps_inv_ptc[:, :-1]

    pts_min_bound = grasps_inv_ptc >= gripper_sweep_volume[:1].unsqueeze(-1)
    pts_max_bound = grasps_inv_ptc <= gripper_sweep_volume[1:].unsqueeze(-1)

    pts_in_bound = pts_max_bound & pts_min_bound
    pts_in_bound = torch.all(pts_in_bound, dim=1)
    mask = ~torch.all(~pts_in_bound, dim=-1)

    if mask.sum().item() > 0:
        mask = mask.numpy()
    else:
        mask = None
    return mask


def load_onpolicy_dataset(
    key: str,
    onpolicy_h5_path: str,
    onpolicy_json_path: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (positive_grasps, negative_grasps, object_scale).
    object_scale is read from the eval-output JSON; None when unavailable."""
    logger.info(f"Loading onpolicy dataset for {key}: {onpolicy_json_path}")

    h5 = h5py.File(onpolicy_h5_path, "r")
    try:
        h5_obj = h5["objects"][key]
    except KeyError:
        object_id = key.split("/", 1)[1] if "/" in key else key
        h5_obj = h5["objects"][object_id]

    json_path = onpolicy_json_path

    pred_grasps = h5_obj["pred_grasps"][...]
    scores = h5_obj["confidence"][...]
    collision = h5_obj["collision"][...]

    logger.info(
        f"ONPOLICY H5 RAW: key={key} pred_grasps={pred_grasps.shape} "
        f"scores={scores.shape} collision={collision.shape}"
    )

    mask_not_colliding = np.logical_not(collision)
    num_grasps_attempted_inference = mask_not_colliding.sum()

    try:
        data = json.load(open(json_path, "rb"))
        num_grasps_attempted_igg = len(data["grasps"]["transforms"])
    except:
        logger.error(f"ONPOLICY: Error in opening file {key}: {onpolicy_json_path}")
        return None, None, None

    onpolicy_object_scale = None
    try:
        onpolicy_object_scale = float(data["object"]["scale"])
    except (KeyError, TypeError, ValueError):
        pass
    # pred_grasps2 = np.array(data['grasps']['transforms'])
    mask_eval_success = np.array(data["grasps"]["object_in_gripper"])
    n_eval = len(mask_eval_success)
    num_total = len(scores)
    num_not_colliding = int(num_grasps_attempted_inference)
    success_result = np.zeros(num_total)
    nc_indices = np.where(mask_not_colliding)[0]

    if n_eval <= num_not_colliding:
        successful_positions = np.where(mask_eval_success)[0]
        success_result[nc_indices[successful_positions]] = 1.0
        if n_eval < num_not_colliding:
            logger.warning(
                f"ONPOLICY {key}: partial eval — {n_eval}/{num_not_colliding} "
                f"non-colliding grasps evaluated."
            )
    elif n_eval <= num_total:
        # Post-patch: collision labels corrected after eval ran.
        # Pre-patch nearly all grasps were non-colliding, so eval
        # index ≈ global grasp index.
        valid = nc_indices < n_eval
        valid_nc = nc_indices[valid]
        eval_for_nc = mask_eval_success[valid_nc]
        success_result[valid_nc[eval_for_nc.astype(bool)]] = 1.0
        n_skipped = int((~valid).sum())
        logger.warning(
            f"ONPOLICY {key}: post-patch collision detected "
            f"(non-colliding={num_not_colliding}, eval={n_eval}). "
            f"Mapped {int(valid.sum())} grasps"
            + (f", skipped {n_skipped} out-of-range." if n_skipped else ".")
        )
    else:
        logger.error(
            f"ONPOLICY {key}: eval results ({n_eval}) exceed total grasps "
            f"({num_total}). Skipping."
        )
        return None, None, None

    positive_grasps_onpolicy = pred_grasps[success_result.astype(bool)]
    negative_grasps_onpolicy = pred_grasps[~success_result.astype(bool)]
    logger.info(
        f"ONPOLICY: num pos {len(positive_grasps_onpolicy)}, num neg: {len(negative_grasps_onpolicy)}"
    )

    return positive_grasps_onpolicy, negative_grasps_onpolicy, onpolicy_object_scale
