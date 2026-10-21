# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference script for GraspGen.
"""

import os
from functools import partial
from pathlib import Path
from time import time

import h5py
import shutil
import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from scene_synthesizer.scene import Scene
import trimesh.transformations as tra
from omegaconf import DictConfig
from scene_synthesizer.assets import Asset, BoxAsset, TrimeshAsset

from graspgenx.dataset.eval_utils import (
    check_collision,
    get_logger,
    get_timestamp,
    log_worker,
    write_info,
    write_to_h5,
)
from graspgenx.utils.training import get_xgrasp_data_loader, to_cpu, to_gpu
from graspgenx.utils.compute_utils import log_all_resources
from graspgenx.utils.logging_config import get_logger as get_module_logger
from graspgenx.x_grippers import resolve_gripper_info
from graspgenx.models.grasp_gen import find_the_last_ckpt

logger = get_module_logger(__name__)

from scripts.curate_ord_eval_chunks import split_h5_file


def load_scene(inputs):
    scene = Scene()
    if "table" in inputs:
        table = BoxAsset(inputs["table"]["size"])
        transform = tra.translation_matrix(inputs["table"]["pos"])
        scene.add_object("table", table, transform)
    if "robot_table" in inputs:
        table = BoxAsset(inputs["robot_table"]["size"])
        transform = tra.translation_matrix(inputs["robot_table"]["pos"])
        scene.add_object("robot_table", table, transform)
    if "robot" in inputs:
        robot = Asset(
            f"{Path(__file__).parent.parent}/assets/franka/franka_panda.urdf",
            configuration=inputs["robot"]["config"],
        )
        scene.add_object("robot", robot, inputs["robot"]["pose"])
    return scene


def record_worker(rank, queue, log_queue, record_fn, args, output_file):
    log = get_logger(f"Worker {rank:02d}", log_queue)
    log.info("Started")
    while True:
        inputs = queue.get()
        if inputs is None:
            break
        record_fn(log, args, inputs, output_file)
        queue.task_done()


def record_grasps_diffusion(log, cfg, inputs, output_file=None):
    args = cfg.eval
    if args.debug and output_file is None:
        pass

    start = time()
    scene = load_scene(inputs["scene_info"])
    asset = inputs["scene_info"].pop("assets")[0]
    scale = inputs["scene_info"].pop("scales")[0]
    pose = inputs["scene_info"].pop("poses")[0]
    gripper = inputs["scene_info"].pop("grippers")[0]
    gripper = resolve_gripper_info(gripper)
    gripper_mesh = gripper.collision_mesh
    mesh_asset = Asset(f"{asset}", scale=scale)

    num_grasps = inputs["grasps_pred"].shape[0]
    all_grasps = inputs["grasps_pred"].cpu().numpy()
    grasps_gt = inputs["gt_grasps"]

    if "labels" in inputs:
        labels = inputs["labels"].cpu().numpy()
        label_mask = np.where(labels.flatten())[0]
        grasps_gt = grasps_gt[label_mask]

    if "confidence" in inputs:
        all_conf = inputs["confidence"].cpu().numpy()
        if len(all_conf.shape) == 2:
            all_conf = all_conf.squeeze(1)
    else:
        all_conf = np.ones(num_grasps)

    if "likelihood" in inputs:
        all_likelihood = inputs["likelihood"].cpu().numpy()
        if len(all_likelihood.shape) == 2:
            all_likelihood = all_likelihood.squeeze(1)

    all_grasps = inputs["cam_pose"] @ all_grasps
    T_move_back_to_obj_frame = tra.inverse_matrix(pose)

    grasps_output = {}
    grasps_output["obj0"] = {
        "pred_grasps": T_move_back_to_obj_frame @ all_grasps,
        "confidence": all_conf,
        "likelihood": all_likelihood,
        "gt_grasps": T_move_back_to_obj_frame @ inputs["cam_pose"] @ grasps_gt,
    }

    try:
        scene.add_object(asset=mesh_asset, obj_id="obj0", transform=np.eye(4))
    except:
        print("Scenethysizer Error.")

    scene_mesh = scene.scene.dump(concatenate=True)
    collision = check_collision(
        scene_mesh, gripper.collision_mesh, grasps_output["obj0"]["pred_grasps"]
    )

    collision_rate = collision.mean() if num_grasps > 0 else np.nan
    all_col = collision.copy()

    grasps_output["obj0"]["collision"] = all_col

    log.info(
        f"Scene {inputs['scene']} Collision rate {collision_rate} "
        f"Total {num_grasps} Average {num_grasps} grasps"
    )

    if args.debug and output_file is None:
        grasps_gt = grasps_output["obj0"]["gt_grasps"]
        grasps_pred = grasps_output["obj0"]["pred_grasps"]
        grasps_per_iteration = inputs["grasps_per_iteration"].cpu().numpy()

        # Plotting args
        step_diffusion_iterations = False
        plot_reverse = True
        plot_thresholded = True
        plot_mesh = False
        use_likelihood_as_score = True

        if use_likelihood_as_score:
            scores = all_likelihood
            score_range = scores.max() - scores.min()
            scores = (scores - scores.min()) / score_range
            scores = get_color_from_score(scores, use_255_scale=True)
        else:
            scores = get_color_from_score(all_conf.cpu().numpy(), use_255_scale=True)

        print(f"Confidence, max: {all_conf.max()}, min {all_conf.min()}")
        print(f"Likelihood, max: {all_likelihood.max()}, min {all_likelihood.min()}")

        if plot_thresholded:
            threshold = 0.6
            if all_conf.max() <= threshold:
                threshold = max(0, all_conf.max() - 0.10)
            mask_thresh = all_conf > threshold

            grasps_visualized = grasps_pred[mask_thresh]
            print(
                f"Thresholding grasps at {threshold}. {grasps_visualized.shape[0]}/{all_grasps.shape[0]} grasps remaining to visualize"
            )

            visualize_mesh(vis, "scene_mesh", scene_mesh, color=[192, 192, 192])

            for j, grasp in enumerate(
                grasps_visualized[: min(len(grasps_visualized), 20)]
            ):
                visualize_x_grasp(
                    vis,
                    f"pred_thresholded/grasp_{j:03d}",
                    grasp,
                    color=[250, 0, 250],
                    gripper=gripper,
                    linewidth=3.0,
                )
                if j < 10:
                    if plot_mesh:
                        visualize_mesh(
                            vis,
                            f"pred_thresholded_meshes/grasp_{j:03d}",
                            gripper_mesh,
                            color=[240, 0, 150],
                            transform=grasp,
                        )

        for j, grasp in enumerate(grasps_gt[:20]):
            visualize_x_grasp(
                vis,
                f"gt/grasp_{j:03d}",
                grasp,
                [0, 250, 0],
                gripper=gripper,
                linewidth=2.0,
            )

        if step_diffusion_iterations:
            timesteps = list(range(len(grasps_per_iteration)))
            if plot_reverse:
                timesteps.reverse()
            for t in timesteps:
                print(t)
                grasps_t = grasps_per_iteration[t]
                for j, g in enumerate(grasps_t):
                    if j < 10:
                        visualize_mesh(
                            vis,
                            f"meshes_t/grasp_{j:03d}",
                            gripper_mesh,
                            color=[0, 150, 250],
                            transform=T_move_back_to_obj_frame @ g.astype(np.float),
                        )
                input()

        input()

    print(f"[DEBUG] output_file: {output_file}")
    if output_file is not None:
        key_id = inputs["scene"]

        saved_data_dict = {}

        from graspgenx.dataset.xgrasp_dataset_utils import (
            XGraspJsonDatasetReader,
            load_object_xgrasp_data,
        )

        if cfg.eval.grasp_split is None:
            grasp_split = cfg.eval.obj_split
        else:
            grasp_split = cfg.eval.grasp_split

        grasp_dataset_reader = XGraspJsonDatasetReader(
            f"{cfg.data.grasp_root_dir}/{grasp_split}",
            cfg.data.object_root_dir,
            alternative_json_file_path=cfg.data.get("alternative_json_file_path", None),
        )

        error_code, object_grasp_data = load_object_xgrasp_data(
            key_id,
            cfg.data.object_root_dir,
            f"{cfg.data.grasp_root_dir}/{grasp_split}",
            min_grasps_gen=cfg.data.min_grasps_gen_th,
            load_discriminator_dataset=cfg.data.load_discriminator_dataset,
            grasp_dataset_reader=grasp_dataset_reader,
        )

        if object_grasp_data is not None:
            asset_path_rel = object_grasp_data.object_asset_path.split("/")[-1]

            print(f"[DEBUG] asset_path_rel: {asset_path_rel}")
            saved_data_dict["asset_path"] = asset_path_rel
            saved_data_dict["asset_scale"] = object_grasp_data.object_scale

            saved_data_dict.update(grasps_output["obj0"])
            print(
                f"Pred {saved_data_dict['pred_grasps'].shape}, Gt {saved_data_dict['gt_grasps'].shape} grasp number"
            )

            grp = output_file.create_group(key_id)
            start = time()
            write_info(grp, saved_data_dict)
            end = time()
            print(f"Writing scene {key_id} data took", end - start, "s")

    time_taken = time() - start
    log.info(
        f"Scene {inputs['scene']} took {round(time_taken,2)} s, saved to {output_file}"
    )


def split_gripper_chunks(grasp_dir, max_grasps=100):

    with h5py.File(f"{grasp_dir}/x_grippers.h5", "r") as f_in:
        for gripper in f_in["objects"].keys():
            with h5py.File(f"{grasp_dir}/{gripper}.h5", "w") as f_out:
                # Copy misc group
                f_in.copy("misc", f_out)
                f_out["misc"]["gripper_name"][()] = gripper

                # Create new 'objects' group and populate it
                f_in.copy(f"objects/{gripper}", f_out, "objects")

            split_h5_file(
                f"{grasp_dir}/{gripper}.h5", f"{grasp_dir}/{gripper}", max_grasps
            )


@hydra.main(config_path=".", config_name="config_xgrasp", version_base="1.3")
def main(cfg: DictConfig) -> None:

    try:
        log_all_resources(logger, tag="inference_startup", include_gpu=True)
    except Exception as e:
        logger.warning(f"log_all_resources failed (non-fatal): {e}")

    sampler, loader = get_xgrasp_data_loader(
        cfg.eval,
        cfg.data,
        cfg.eval.obj_split,
        cfg.eval.gripper_split,
        cfg.eval.grasp_split,
        use_ddp=False,
        use_cache=True,
        training=False,
        inference=True,
    )

    # Per-gripper filtering (no-op when single_gripper is null/None)
    single_gripper = cfg.eval.get("single_gripper", None)
    if single_gripper is not None:
        original_count = len(loader.dataset.scenes)
        loader.dataset.scenes = [
            s for s in loader.dataset.scenes if s.startswith(f"{single_gripper}/")
        ]
        print(
            f"[Per-gripper mode] Filtered to gripper '{single_gripper}': "
            f"{len(loader.dataset.scenes)}/{original_count} scenes"
        )

    from graspgenx.models.grasp_gen import GraspGen, GraspGenGenerator

    if cfg.eval.model_name == "diffusion":
        model = GraspGenGenerator.from_config(cfg.diffusion)
    elif cfg.eval.model_name == "diffusion-discriminator":
        model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
    else:
        raise NotImplementedError(f"Model name not implemented {cfg.eval.model_name}")

    if cfg.eval.model_name == "diffusion-discriminator":
        model.load_state_dict(cfg.eval.gen_checkpoint, cfg.eval.dis_checkpoint)
    elif cfg.eval.model_name == "diffusion":
        try:
            ckpt = torch.load(cfg.eval.gen_checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model"])
        except:
            # if the last ckpt is corrupted, find the one with the largest epoch num
            ckpt = find_the_last_ckpt(cfg.eval.gen_checkpoint)
            model.load_state_dict(ckpt["model"])

    model = model.cuda().eval()

    if cfg.eval.output_dir is not None:
        os.makedirs(cfg.eval.output_dir, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    log_queue = mp.Queue()
    log_proc = mp.Process(target=log_worker, args=(log_queue,))
    log_proc.start()
    log = get_logger("main", log_queue)

    out_dir = f"{cfg.eval.output_dir}/{cfg.eval.exp_name}"
    os.makedirs(out_dir, exist_ok=True)

    if single_gripper is not None:
        h5_file_name = f"{single_gripper}.h5"
        tmp_file_name = f"{single_gripper}_tmp.h5"
    else:
        h5_file_name = "x_grippers.h5"
        tmp_file_name = "x_grippers_tmp.h5"

    output_file_path = os.path.join(out_dir, h5_file_name)
    log.info(f"Saving to {output_file_path}")

    tmp_file_path = os.path.join(out_dir, tmp_file_name)
    os.system(f"rm {tmp_file_path}")  # For safety

    if os.path.exists(output_file_path):
        shutil.copyfile(output_file_path, tmp_file_path)
        output_file = h5py.File(tmp_file_path, "a")
        h5_handle = output_file["objects"]
    else:
        output_file = h5py.File(tmp_file_path, "a")
        misc_data = {"model": cfg.eval.exp_name, "gripper_name": "x_grippers"}
        grp = output_file.create_group("misc")
        write_info(grp, misc_data)
        h5_handle = output_file.create_group("objects")

    # Evaluate all grasps
    log.info(f"Evaluating all grasps without thresholding")
    cfg.eval.object_thresh = -1.0
    cfg.eval.mask_thresh = -1.0

    record_fn = record_grasps_diffusion
    if not cfg.eval.debug:
        queue = mp.JoinableQueue()
        procs = [
            mp.Process(
                target=record_worker,
                args=(i, queue, log_queue, record_fn, cfg, h5_handle),
            )
            for i in range(cfg.eval.num_procs)
        ]
        for p in procs:
            p.start()

    data_time, infer_time, record_time, num_scenes = 0, 0, 0, 0
    start = time()
    for i, data in enumerate(loader):
        if data is None:
            continue

        if data["scene"][0] in h5_handle:
            print(f"Key {data['scene'][0]} exists. Continue.")
            continue

        to_gpu(data)
        num_scenes += len(data["scene"])
        data_time += time() - start

        start = time()
        with torch.no_grad():
            outputs, _, stats = model.infer(data, return_metrics=True)

        print(
            f"Stats, L2 error:{stats['error_trans_l2'].item()} recall: {stats['recall'].item()} phi3 {stats['error_rot_phi3'].item()} "
        )

        infer_time += time() - start
        start = time()

        to_cpu(data)
        to_cpu(outputs)

        for j in range(len(data["scene"])):
            if cfg.eval.cam_coord:
                cam_pose = data["cam_pose"][j].numpy()
            else:
                cam_pose = np.eye(4)

            inputs = {
                "scene": data["scene"][j],
                "scene_info": data["scene_info"][j],
                "cam_pose": cam_pose,
            }

            if cfg.eval.task == "pick":
                inputs.update(
                    {
                        "gt_grasps": data["grasps_ground_truth"][j],
                        "grasps_pred": outputs["grasps_pred"][j],
                        "likelihood": outputs["likelihood"][j],
                        "grasps_per_iteration": outputs["grasps_per_iteration"][j],
                        "confidence": outputs["grasp_confidence"][j],
                        "grasping_masks": outputs["grasping_masks"][j],
                        "contacts": outputs["grasp_contacts"][j],
                    }
                )

                # For GraspGenGenerator
                for key in ["seg", "rgb", "xyz", "points", "bbox", "labels"]:
                    if key in data:
                        inputs.update({key: data[key][j]})

            if cfg.eval.debug:
                record_fn(log, cfg, inputs, h5_handle)
            else:
                queue.put(inputs)

        record_time += time() - start
        start = time()
        if (i + 1) % cfg.eval.print_freq == 0:
            log.info(
                f"{i+1}/{len(loader)} "
                f"Data time {data_time / num_scenes} "
                f"Inference time {infer_time / num_scenes} "
                f"Record time {record_time / num_scenes} "
                f"num scenes {num_scenes}"
            )
            data_time, infer_time, record_time, num_scenes = 0, 0, 0, 0

        # Early stopping for debugging: stop after N objects
        max_eval = cfg.eval.get("max_eval_objects", -1)
        if max_eval > 0 and (i + 1) >= max_eval:
            log.info(f"Reached max_eval_objects={max_eval}, stopping early.")
            break

        if (i + 1) % 5000 == 0:
            output_file.close()  # save once in a while
            shutil.move(tmp_file_path, output_file_path)
            shutil.copyfile(output_file_path, tmp_file_path)
            output_file = h5py.File(tmp_file_path, "a")
            h5_handle = output_file["objects"]

    if not cfg.eval.debug:
        queue.join()
        print("All work completed")
        for _ in range(cfg.eval.num_procs * 2):
            queue.put(None)
        for p in procs:
            p.join()

    log_queue.put(None)
    log_proc.join()

    # separate into different
    output_file.close()
    shutil.move(tmp_file_path, output_file_path)

    if single_gripper is not None:
        # Per-gripper: flatten objects/{gripper}/{obj} → objects/{obj}, then chunk
        flat_h5 = os.path.join(out_dir, f"{single_gripper}_flat.h5")
        with h5py.File(output_file_path, "r") as f_in:
            with h5py.File(flat_h5, "w") as f_out:
                f_in.copy("misc", f_out)
                f_out["misc"]["gripper_name"][()] = single_gripper
                f_in.copy(f"objects/{single_gripper}", f_out, "objects")
        os.replace(flat_h5, output_file_path)
        split_h5_file(
            output_file_path,
            os.path.join(out_dir, single_gripper),
            cfg.eval.max_grasps_per_chunk,
        )
    else:
        split_gripper_chunks(out_dir, cfg.eval.max_grasps_per_chunk)


if __name__ == "__main__":
    main()
