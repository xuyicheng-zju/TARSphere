# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Checkpoint discovery and model-config assembly for the demo scripts.

import glob
import os
from pathlib import Path

import omegaconf


def find_latest_checkpoint(ckpt_dir: str) -> str:
    """Return the .pth file with the highest epoch number in ``ckpt_dir``.

    Prefers files named ``epoch_<N>.pth``; falls back to any ``*.pth`` if no
    epoch-tagged file is present. Raises FileNotFoundError if the directory
    has no .pth files at all.
    """
    pth_files = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if not pth_files:
        pth_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth checkpoint files found in {ckpt_dir}")

    def epoch_number(path):
        stem = Path(path).stem
        if stem.startswith("epoch_"):
            try:
                return int(stem.split("_")[1])
            except (IndexError, ValueError):
                pass
        return -1

    pth_files.sort(key=epoch_number)
    return pth_files[-1]


def load_model_cfg(
    gen_dir: str, dis_dir: str, gen_pth=None, dis_pth=None
) -> omegaconf.DictConfig:
    """Build a merged GraspGenX config from separate gen/dis checkpoint dirs.

    The discriminator config is used as the base; the generator's ``diffusion``
    block is spliced in so the model loads both backbones correctly. Auto-picks
    the latest epoch when ``gen_pth`` / ``dis_pth`` are ``None``.
    """
    gen_dir = str(Path(gen_dir).resolve())
    dis_dir = str(Path(dis_dir).resolve())

    gen_cfg_path = os.path.join(gen_dir, "config.yaml")
    dis_cfg_path = os.path.join(dis_dir, "config.yaml")

    if not os.path.exists(gen_cfg_path):
        raise FileNotFoundError(f"Generator config not found: {gen_cfg_path}")
    if not os.path.exists(dis_cfg_path):
        raise FileNotFoundError(f"Discriminator config not found: {dis_cfg_path}")

    dis_cfg = omegaconf.OmegaConf.load(dis_cfg_path)
    gen_cfg = omegaconf.OmegaConf.load(gen_cfg_path)

    cfg = dis_cfg
    cfg.diffusion = gen_cfg.diffusion

    if gen_pth is None:
        gen_pth = find_latest_checkpoint(gen_dir)
    else:
        gen_pth = os.path.join(gen_dir, gen_pth)

    if dis_pth is None:
        dis_pth = find_latest_checkpoint(dis_dir)
    else:
        dis_pth = os.path.join(dis_dir, dis_pth)

    cfg.eval.gen_checkpoint = gen_pth
    cfg.eval.dis_checkpoint = dis_pth

    print(f"Generator checkpoint : {cfg.eval.gen_checkpoint}")
    print(f"Discriminator checkpoint: {cfg.eval.dis_checkpoint}")

    gen = cfg.diffusion
    dis = cfg.discriminator
    print(
        f"Generator      : backbone={gen.object_backbone}, gripper={gen.gripper_backbone}, "
        f"obj_dim={gen.num_object_dim}, embed_dim={gen.diffusion_embed_dim}, "
        f"pointnet_ver={gen.pointnet_version}"
    )
    print(
        f"  DDPM         : train_steps={gen.num_diffusion_iters}, "
        f"eval_steps={gen.num_diffusion_iters_eval}, beta_schedule={gen.beta_schedule}, "
        f"clip_sample={gen.clip_sample}, compositional={gen.compositional_schedular}"
    )
    print(
        f"  Grasp        : grasp_repr={gen.grasp_repr}, kappa(noise_scale)={gen.kappa}, "
        f"pose_repr={gen.pose_repr}"
    )
    print(
        f"Discriminator  : backbone={dis.object_backbone}, gripper={dis.gripper_backbone}, "
        f"obj_dim={dis.num_object_dim}, embed_dim={dis.num_embed_dim}, "
        f"pointnet_ver={dis.pointnet_version}"
    )
    print(
        f"  Grasp        : grasp_repr={dis.grasp_repr}, kappa(noise_scale)={dis.kappa}, "
        f"pose_repr={dis.pose_repr}"
    )

    return cfg
