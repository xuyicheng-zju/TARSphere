#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect evaluation H5 files and optionally cross-reference with eval_output JSON results.

Usage
-----

1) Inspect a single H5 file (per-object grasp details):

    python scripts/inspect_eval_h5.py path/to/gripper.h5

2) Inspect a directory of H5 files (summary table of all grippers):

    python scripts/inspect_eval_h5.py path/to/eval_input/<exp_name>/

   The --dir flag is accepted but optional; a directory path is auto-detected.

3) Inspect a directory AND cross-reference with Isaac Sim evaluation results:

    python scripts/inspect_eval_h5.py path/to/eval_input/<exp_name>/ \\
        --json_dir path/to/eval_output/<exp_name>/

   This reports, per gripper: number of objects, predicted grasps,
   non-colliding grasps, how many objects have been evaluated in sim,
   and the grasp success rate.

4) Filter by gripper prefix (works with both directory and directory+json modes):

    python scripts/inspect_eval_h5.py path/to/eval_input/<exp_name>/ \\
        --json_dir path/to/eval_output/<exp_name>/ \\
        --prefix revolute_2f

5) Full example with typical paths:

    python scripts/inspect_eval_h5.py \\
        ~/ord-mount-beiningh/x-grasp-result/eval_input/x_grasp_train_graspgenX_proc_v1_train_32_xgrasp_v1_largered_v2_train_graspgenX_proc_v1_train_32_valid_graspgenX_valid_gripper_cond:sweep_volume_v2_obj:pointnet_pc:0.5_seed:0_gen_multinode_diffusion/ \\
        --json_dir ~/ord-mount-beiningh/x-grasp-result/eval_output/x_grasp_train_graspgenX_proc_v1_train_32_xgrasp_v1_largered_v2_train_graspgenX_proc_v1_train_32_valid_graspgenX_valid_gripper_cond:sweep_volume_v2_obj:pointnet_pc:0.5_seed:0_gen_multinode_diffusion/

H5 structure (eval_input)
-------------------------
Per-gripper files:  <gripper_name>.h5  ->  objects/<object_id>/{pred_grasps, gt_grasps, confidence, collision, ...}
Combined file:      x_grippers.h5      ->  objects/<gripper_name>/<object_id>/{...}

JSON structure (eval_output)
----------------------------
<gripper_name>/chunk_*/*_<object_id>/grasps.json
  -> grasps.transforms        (list of 4x4 transforms sent to sim)
  -> grasps.object_in_gripper (bool list: True = grasp succeeded)

Only non-colliding grasps (collision==False in H5) are sent to Isaac Sim,
so len(grasps.transforms) should equal the non-colliding count from the H5.
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
from tqdm import tqdm

_MAX_WORKERS = 16


def inspect_h5(path):
    """Print detailed per-object grasp info for a single H5 file."""
    with h5py.File(path, "r") as f:
        if "misc" in f:
            misc = f["misc"]
            print("=== Misc ===")
            for k in misc.keys():
                val = misc[k][()]
                if isinstance(val, bytes):
                    val = val.decode()
                print(f"  {k}: {val}")
            print()

        if "objects" not in f:
            print("No 'objects' group found in this file.")
            return

        objects = f["objects"]
        object_keys = list(objects.keys())
        total_objects = len(object_keys)

        print(f"=== Objects: {total_objects} ===")

        total_pred_grasps = 0
        total_gt_grasps = 0

        for key in tqdm(sorted(object_keys), desc="Objects", unit="obj"):
            grp = objects[key]
            pred = grp["pred_grasps"].shape[0] if "pred_grasps" in grp else 0
            gt = grp["gt_grasps"].shape[0] if "gt_grasps" in grp else 0
            total_pred_grasps += pred
            total_gt_grasps += gt

            extras = []
            if "confidence" in grp:
                conf = grp["confidence"][:]
                extras.append(f"conf=[{conf.min():.3f}, {conf.max():.3f}]")
            if "collision" in grp:
                col = grp["collision"][:]
                extras.append(f"collision={col.mean():.3f}")

            extra_str = f"  ({', '.join(extras)})" if extras else ""
            tqdm.write(f"  {key}: pred={pred}, gt={gt}{extra_str}")

        print()
        print(f"=== Summary ===")
        print(f"  Total objects:          {total_objects}")
        print(f"  Total pred grasps:      {total_pred_grasps}")
        print(f"  Total gt grasps:        {total_gt_grasps}")
        if total_objects > 0:
            print(f"  Avg pred grasps/obj:    {total_pred_grasps / total_objects:.1f}")
            print(f"  Avg gt grasps/obj:      {total_gt_grasps / total_objects:.1f}")


def _count_h5_objects(h5_path):
    """Count objects in a single H5 file. Returns (filename, count, error_or_None)."""
    fname = os.path.basename(h5_path)
    try:
        with h5py.File(h5_path, "r") as f:
            n_objects = len(f["objects"].keys()) if "objects" in f else 0
        return (fname, n_objects, None)
    except Exception as e:
        return (fname, 0, str(e))


def inspect_dir(directory, prefix=None):
    """Find all .h5 files in a directory and print object counts for each."""
    h5_files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    if prefix:
        h5_files = [
            p
            for p in h5_files
            if os.path.basename(p)
            .replace(".h5", "")
            .replace("_tmp", "")
            .startswith(prefix)
        ]
    if not h5_files:
        print(
            f"No .h5 files found in {directory}"
            + (f" matching prefix '{prefix}'" if prefix else "")
        )
        return

    print(
        f"=== Found {len(h5_files)} H5 files in {directory}"
        + (f" (prefix='{prefix}')" if prefix else "")
        + " ===\n"
    )
    print(f"{'File':<50s} {'Objects':>8s}")
    print("-" * 60)

    max_workers = min(_MAX_WORKERS, len(h5_files))
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_count_h5_objects, p): p for p in h5_files}
        for future in tqdm(
            as_completed(future_to_path),
            total=len(h5_files),
            desc="Reading H5 files",
            unit="file",
        ):
            results[future_to_path[future]] = future.result()

    total_objects = 0
    for h5_path in h5_files:
        fname, n_objects, err = results[h5_path]
        if err is not None:
            print(f"  {fname:<48s} {'ERROR':>8s}  ({err})")
        else:
            total_objects += n_objects
            print(f"  {fname:<48s} {n_objects:>8d}")

    print("-" * 60)
    print(f"  {'TOTAL':<48s} {total_objects:>8d}")
    print(f"  Files: {len(h5_files)}")


# ---------------------------------------------------------------------------
# JSON cross-referencing helpers
# ---------------------------------------------------------------------------


def _find_eval_json(json_dir, gripper_name, object_id):
    """Find the grasps.json for a given gripper/object in eval_output."""
    pattern = os.path.join(
        json_dir, gripper_name, "chunk_*", f"*_{object_id}", "grasps.json"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _read_eval_json(json_path):
    """Return (n_evaluated, n_successful) from a grasps.json, or None on error."""
    try:
        with open(json_path, "r") as jf:
            data = json.load(jf)
        mask = np.array(data["grasps"]["object_in_gripper"])
        return int(len(mask)), int(mask.sum())
    except Exception as e:
        print(f"    WARNING: Error reading {json_path}: {e}")
        return None


def _process_gripper_objects(objects_group, json_dir, gripper_name):
    """Aggregate stats for all objects under a gripper group in an H5 file.

    Returns (stats_dict, missing_object_ids) where missing_object_ids is a list
    of object IDs that have non-colliding grasps but no eval JSON result.
    """
    object_ids = sorted(objects_group.keys())
    stats = dict(
        n_objects=len(object_ids),
        n_pred=0,
        n_non_colliding=0,
        n_evaluated_objects=0,
        n_evaluated_grasps=0,
        n_successful=0,
    )
    missing = []

    for obj_id in tqdm(
        object_ids, desc=f"  {gripper_name} objects", unit="obj", leave=False
    ):
        grp = objects_group[obj_id]
        n_pred = grp["pred_grasps"].shape[0] if "pred_grasps" in grp else 0
        stats["n_pred"] += n_pred

        collision = (
            grp["collision"][:] if "collision" in grp else np.zeros(n_pred, dtype=bool)
        )
        n_nc = int((~collision).sum())
        stats["n_non_colliding"] += n_nc

        evaluated = False
        if json_dir is not None:
            jp = _find_eval_json(json_dir, gripper_name, obj_id)
            if jp is not None:
                result = _read_eval_json(jp)
                if result is not None:
                    stats["n_evaluated_objects"] += 1
                    stats["n_evaluated_grasps"] += result[0]
                    stats["n_successful"] += result[1]
                    evaluated = True

        if json_dir is not None and not evaluated and n_nc > 0:
            missing.append(obj_id)

    return stats, missing


def _print_gripper_stats(gripper_name, s, label_suffix=""):
    """Pretty-print a single gripper's stats dict."""
    tag = f"{gripper_name}{label_suffix}"
    eval_pct = s["n_evaluated_objects"] / s["n_objects"] * 100 if s["n_objects"] else 0
    sr = (
        s["n_successful"] / s["n_evaluated_grasps"] * 100
        if s["n_evaluated_grasps"]
        else 0
    )

    print(f"--- {tag} ---")
    print(
        f"  Objects:      {s['n_objects']:>6d}  |  Evaluated: {s['n_evaluated_objects']}/{s['n_objects']} ({eval_pct:.1f}%)"
    )
    print(
        f"  Pred grasps:  {s['n_pred']:>6d}  |  Non-colliding: {s['n_non_colliding']}"
    )
    print(
        f"  Eval grasps:  {s['n_evaluated_grasps']:>6d}  |  Successful: {s['n_successful']}  |  Success rate: {sr:.1f}%"
    )


def _process_single_h5(h5_path, json_dir, prefix):
    """Process one H5 file, returning gripper stats without printing.

    Returns a list of (gripper_name, stats_dict, label_suffix, missing_objects) tuples,
    or a string message (error / skip).
    """
    fname = os.path.basename(h5_path)
    gripper_name = fname.replace(".h5", "").replace("_tmp", "")

    try:
        with h5py.File(h5_path, "r") as f:
            if "objects" not in f:
                return f"--- {gripper_name}: No 'objects' group ---"

            first_key = next(iter(f["objects"].keys()))
            is_combined = (
                isinstance(f["objects"][first_key], h5py.Group)
                and "pred_grasps" not in f["objects"][first_key]
            )

            entries = []
            if is_combined:
                for sub_gripper in sorted(f["objects"].keys()):
                    if prefix and not sub_gripper.startswith(prefix):
                        continue
                    s, missing = _process_gripper_objects(
                        f["objects"][sub_gripper], json_dir, sub_gripper
                    )
                    entries.append((sub_gripper, s, f" (from {fname})", missing))
            else:
                s, missing = _process_gripper_objects(
                    f["objects"], json_dir, gripper_name
                )
                entries.append((gripper_name, s, "", missing))

            return entries
    except Exception as e:
        return f"--- {gripper_name}: ERROR ({e}) ---"


def _load_obj_split(obj_split_file):
    """Read object IDs from a text file (one per line)."""
    ids = []
    with open(obj_split_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def find_missing_eval_objects(directory, obj_split_file, prefix=None, output_json=None):
    """Compare each gripper's H5 object set against the full object split.

    Prints per-gripper counts and top-5 missing IDs, and optionally writes
    the full mapping to a JSON file.
    """
    obj_split = _load_obj_split(obj_split_file)
    obj_set = set(obj_split)

    h5_files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    if prefix:
        h5_files = [
            p
            for p in h5_files
            if os.path.basename(p)
            .replace(".h5", "")
            .replace("_tmp", "")
            .startswith(prefix)
        ]
    if not h5_files:
        print(
            f"No .h5 files found in {directory}"
            + (f" matching prefix '{prefix}'" if prefix else "")
        )
        return

    print(f"=== Object split: {obj_split_file}  ({len(obj_split)} objects) ===")
    print(f"=== H5 dir: {directory} ===\n")

    all_missing = {}
    total_missing = 0

    for h5_path in tqdm(h5_files, desc="Checking H5 files", unit="file"):
        fname = os.path.basename(h5_path)
        gripper_name = fname.replace(".h5", "").replace("_tmp", "")
        try:
            with h5py.File(h5_path, "r") as f:
                if "objects" not in f:
                    continue
                first_key = next(iter(f["objects"].keys()))
                is_combined = (
                    isinstance(f["objects"][first_key], h5py.Group)
                    and "pred_grasps" not in f["objects"][first_key]
                )
                if is_combined:
                    for sub_gripper in sorted(f["objects"].keys()):
                        if prefix and not sub_gripper.startswith(prefix):
                            continue
                        h5_objs = set(f["objects"][sub_gripper].keys())
                        missing = sorted(obj_set - h5_objs)
                        if missing:
                            all_missing[sub_gripper] = missing
                        total_missing += len(missing)
                        preview = missing[:5]
                        tqdm.write(
                            f"  {sub_gripper}: {len(missing)} missing"
                            + (
                                f"  (e.g. {', '.join(preview)}{'...' if len(missing) > 5 else ''})"
                                if missing
                                else ""
                            )
                        )
                else:
                    h5_objs = set(f["objects"].keys())
                    missing = sorted(obj_set - h5_objs)
                    if missing:
                        all_missing[gripper_name] = missing
                    total_missing += len(missing)
                    preview = missing[:5]
                    tqdm.write(
                        f"  {gripper_name}: {len(missing)} missing"
                        + (
                            f"  (e.g. {', '.join(preview)}{'...' if len(missing) > 5 else ''})"
                            if missing
                            else ""
                        )
                    )
        except Exception as e:
            tqdm.write(f"  {gripper_name}: ERROR ({e})")

    print(
        f"\n  Total missing object-gripper combos: {total_missing} across {len(all_missing)} grippers"
    )
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as fp:
            json.dump(all_missing, fp, indent=2)
        print(f"  Dumped to: {output_json}")


def inspect_dir_with_json(directory, json_dir, prefix=None, dump_missing=None):
    """Inspect H5 files and cross-reference with eval_output JSON results.

    If dump_missing is a file path, write a JSON mapping of
    gripper -> [object_ids...] for combos that have non-colliding grasps
    but no evaluation result yet.
    """
    h5_files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    if prefix:
        h5_files = [
            p
            for p in h5_files
            if os.path.basename(p)
            .replace(".h5", "")
            .replace("_tmp", "")
            .startswith(prefix)
        ]
    if not h5_files:
        print(
            f"No .h5 files found in {directory}"
            + (f" matching prefix '{prefix}'" if prefix else "")
        )
        return

    print(
        f"=== Found {len(h5_files)} H5 files in {directory}"
        + (f" (prefix='{prefix}')" if prefix else "")
        + " ==="
    )
    print(f"=== JSON eval dir: {json_dir} ===\n")

    max_workers = min(_MAX_WORKERS, len(h5_files))
    all_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_process_single_h5, p, json_dir, prefix): p
            for p in h5_files
        }
        for future in tqdm(
            as_completed(future_to_path),
            total=len(h5_files),
            desc="Processing H5 files",
            unit="file",
        ):
            all_results[future_to_path[future]] = future.result()

    _ACCUM_KEYS = (
        "n_objects",
        "n_pred",
        "n_non_colliding",
        "n_evaluated_objects",
        "n_evaluated_grasps",
        "n_successful",
    )
    totals = dict(n_grippers=0, **{k: 0 for k in _ACCUM_KEYS})
    all_missing = {}

    for h5_path in h5_files:
        result = all_results[h5_path]
        if isinstance(result, str):
            print(f"\n{result}")
            continue
        for gripper_name, s, label_suffix, missing in result:
            _print_gripper_stats(gripper_name, s, label_suffix)
            totals["n_grippers"] += 1
            for k in _ACCUM_KEYS:
                totals[k] += s[k]
            if missing:
                all_missing[gripper_name] = missing

    print()
    print("=" * 70)
    print("GRAND TOTAL")
    print("=" * 70)
    eval_pct = (
        totals["n_evaluated_objects"] / totals["n_objects"] * 100
        if totals["n_objects"]
        else 0
    )
    sr = (
        totals["n_successful"] / totals["n_evaluated_grasps"] * 100
        if totals["n_evaluated_grasps"]
        else 0
    )
    print(f"  Grippers:                {totals['n_grippers']}")
    print(f"  Total objects:           {totals['n_objects']}")
    print(
        f"  Evaluated objects:       {totals['n_evaluated_objects']}/{totals['n_objects']} ({eval_pct:.1f}%)"
    )
    print(f"  Total pred grasps:       {totals['n_pred']}")
    print(f"  Total non-colliding:     {totals['n_non_colliding']}")
    print(f"  Total evaluated grasps:  {totals['n_evaluated_grasps']}")
    print(f"  Total successful:        {totals['n_successful']}")
    print(f"  Overall success rate:    {sr:.1f}%")

    n_missing_combos = sum(len(v) for v in all_missing.values())
    print(
        f"\n  Missing eval combos:     {n_missing_combos} objects across {len(all_missing)} grippers"
    )

    if dump_missing:
        os.makedirs(os.path.dirname(os.path.abspath(dump_missing)), exist_ok=True)
        with open(dump_missing, "w") as fp:
            json.dump(all_missing, fp, indent=2)
        print(f"\n  Dumped missing combos to: {dump_missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect evaluation H5 file(s) and optionally cross-reference with sim eval JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Single H5 file
  %(prog)s path/to/gripper.h5

  # Directory of H5 files (auto-detected)
  %(prog)s path/to/eval_input/<exp>/

  # Directory + sim evaluation results
  %(prog)s path/to/eval_input/<exp>/ --json_dir path/to/eval_output/<exp>/
""",
    )
    parser.add_argument(
        "path", type=str, help="Path to an H5 file or a directory containing H5 files"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default=None,
        help="Path to eval_output/<exp> directory with JSON sim results "
        "(enables per-gripper success rate reporting)",
    )
    parser.add_argument(
        "--dir",
        action="store_true",
        help="Treat path as a directory (auto-detected if path is a directory)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Only show grippers whose name starts with this prefix "
        "(e.g. 'revolute_2f')",
    )
    parser.add_argument(
        "--dump_missing",
        type=str,
        default=None,
        help="Path to write a JSON file mapping gripper -> [object_ids] "
        "for combos with non-colliding grasps but no eval result. "
        "Requires --json_dir.",
    )
    parser.add_argument(
        "--missing_eval_objects",
        type=str,
        default=None,
        help="Path to an object-split .txt file. Compares each gripper's "
        "H5 objects against this split and reports/dumps objects that "
        "are in the split but missing from the H5 (i.e. inference was "
        "never run). Output JSON path is derived by appending "
        "'_missing_eval_objects.json' unless --missing_eval_objects_out "
        "is also given.",
    )
    parser.add_argument(
        "--missing_eval_objects_out",
        type=str,
        default=None,
        help="Explicit output path for the missing-eval-objects JSON. "
        "Defaults to missing_eval_objects.json in the current directory.",
    )
    args = parser.parse_args()

    is_directory = args.dir or os.path.isdir(args.path)

    if args.missing_eval_objects:
        if not is_directory:
            print(
                "ERROR: --missing_eval_objects requires a directory path",
                file=sys.stderr,
            )
            sys.exit(1)
        out_json = args.missing_eval_objects_out or "missing_eval_objects.json"
        find_missing_eval_objects(
            args.path,
            args.missing_eval_objects,
            prefix=args.prefix,
            output_json=out_json,
        )
    elif is_directory:
        if args.json_dir:
            inspect_dir_with_json(
                args.path,
                args.json_dir,
                prefix=args.prefix,
                dump_missing=args.dump_missing,
            )
        else:
            if args.dump_missing:
                print("ERROR: --dump_missing requires --json_dir", file=sys.stderr)
                sys.exit(1)
            inspect_dir(args.path, prefix=args.prefix)
    else:
        if args.json_dir:
            print(
                "WARNING: --json_dir is only used when inspecting a directory, ignoring.",
                file=sys.stderr,
            )
        inspect_h5(args.path)
