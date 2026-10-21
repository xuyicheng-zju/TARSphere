#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training script for GraspGen model.
"""

import gc
import os
import signal
import sys
import threading
from datetime import timedelta
from functools import partial
from itertools import chain
from time import sleep, time

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from graspgenx.dataset.xgrasp_dataset import get_cache_path, get_cache_prefix
from graspgenx.dataset.xgrasp_dataset_utils import XGraspGenDatasetCache
from graspgenx.models.grasp_gen import GraspGenDiscriminator, GraspGenGenerator
from graspgenx.utils.training import (
    add_to_dict,
    build_optimizer,
    compute_iou,
    get_xgrasp_data_loader,
    save_model,
    to_cpu,
    to_gpu,
    write_scalar_ddp,
)
from graspgenx.utils.compute_utils import log_all_resources
from graspgenx.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)

# Global variables
handler_called = False


def _pre_load_shared_caches(cfg):
    """Pre-load train and valid dataset caches for multi-GPU DDP.

    When using mp.spawn with N GPUs, each process would normally load its own
    copy of the ~100-200 GB dataset cache into CPU RAM, causing OOM.  This
    function loads the cache once in the main process *before* any CUDA
    initialisation.  Combined with ``start_method='fork'`` in mp.spawn,
    all N workers inherit the same physical pages via the kernel's
    copy-on-write (COW) mechanism.

    ``gc.disable()`` + ``gc.freeze()`` (called in ``main()`` after this
    function returns) prevent the cyclic garbage collector from scanning
    these objects post-fork, which would dirty their pages and trigger
    COW duplication.  Remaining COW from basic Python refcounting is
    estimated at ~4 GB total across all workers — negligible on a 2 TiB node.

    Note: we intentionally do NOT call ``share_memory_()`` because the
    cache contains ~3.5 M individual arrays; each ``share_memory_()``
    call creates a separate mmap region, exceeding the kernel's
    ``vm.max_map_count`` limit (default 65 536).
    """

    def _load_one(grasp_split, gripper_split, obj_split):
        if grasp_split is None:
            grasp_split = obj_split
        cache_dir = get_cache_path(cfg.data.cache_dir, cfg.data.cache_name)
        cache_token = get_cache_prefix(
            cfg.data.prob_point_cloud, cfg.data.load_discriminator_dataset
        )
        cache_file = f"{grasp_split}_{gripper_split}_{cache_token}"
        if cfg.data.onpolicy_dataset_name is not None:
            cache_file = f"{cfg.data.onpolicy_dataset_name}_{cache_file}"

        cache_load_path = f"{cache_dir}/{cache_file}.h5"
        logger.info(f"Pre-loading cache from {cache_load_path}")

        cache = XGraspGenDatasetCache()
        cache.load_from_h5_file(cache_load_path)
        logger.info(f"Cache ready: {len(cache)} entries")
        return cache

    train_cache = _load_one(
        cfg.train.train_grasp_split,
        cfg.train.train_gripper_split,
        cfg.train.train_obj_split,
    )
    valid_cache = _load_one(
        cfg.train.valid_grasp_split,
        cfg.train.valid_gripper_split,
        cfg.train.valid_obj_split,
    )

    # NOTE: We intentionally do NOT call share_memory_() here.
    # The cache has ~3.5M individual arrays; each share_memory_() call
    # creates a separate mmap region, exceeding the kernel's
    # vm.max_map_count limit (default 65536).
    #
    # Instead, gc.disable() + gc.freeze() (called in main() before fork)
    # prevent the cyclic GC from scanning these objects and dirtying
    # their pages.  Remaining COW from Python refcounting is ~4 GB
    # total across all workers — negligible on a 2 TiB node.

    return (train_cache, valid_cache)


def train_one_epoch(
    loader,
    model,
    optimizer,
    clip_grad,
    writer,
    epoch,
    global_step,
    cfg,
    batch_idx,
    rank,
):
    global handler_called

    ws = 1
    use_ddp = dist.is_available() and cfg.train.num_gpus > 1
    if use_ddp:
        rank = dist.get_rank()
        ws = dist.get_world_size()

    # Derive local CUDA device (matches torch.cuda.set_device in train())
    slurm_localid = os.environ.get("SLURM_LOCALID")
    if slurm_localid is not None:
        device_rank = int(slurm_localid)
    else:
        device_rank = rank

    def signal_handler(sig, _):
        global handler_called
        if not handler_called:
            if sig in [signal.SIGTERM, signal.SIGINT] and rank == 0:
                handler_called = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    params = list(chain(*[x["params"] for x in optimizer.param_groups]))
    num_steps = len(loader)
    data_time = torch.tensor(0.0, device=device_rank)
    step_time = torch.tensor(0.0, device=device_rank)
    log_time = torch.tensor(0.0, device=device_rank)

    start = time()
    num_batch_updates = 0
    recent_grad_norms = []

    for i, data in enumerate(loader):

        if handler_called:
            logger.info(
                f"Saving new checkpoints for rank:{rank} epoch: {epoch} batch index: {i}, global_step: {global_step}"
            )
            save_model(
                epoch - 1,
                model,
                optimizer,
                cfg.train.log_dir,
                use_ddp,
                name="last",
                batch_idx=i,
            )
            if writer is not None:
                writer.flush()
            logger.info("Terminating training due to a interrupt, sayonara!")
            sys.exit(0)

        if dist.is_initialized():
            dist.barrier()

        if data is None:
            continue

        global_step += 1
        to_gpu(data)
        data_time += time() - start
        start = time()

        optimizer.zero_grad()

        if dist.is_initialized():
            dist.barrier()
        outputs, losses, stats = model(data, cfg.train)
        if dist.is_initialized():
            dist.barrier()

        loss = sum([w * v for w, v in losses.values()])

        if dist.is_initialized():
            dist.barrier()
        loss.backward()
        if dist.is_initialized():
            dist.barrier()

        grad_has_inf_nan = False
        if clip_grad is not None:
            # print("Clipping gradients")
            grad_norm = clip_grad(params)
            grad_has_inf_nan = grad_norm.isinf() or grad_norm.isnan()
            if use_ddp:
                dist.reduce(grad_norm, dst=0)
            if rank == 0:
                if grad_norm.isinf():
                    logger.warning(
                        "Epoch", epoch, "Step", i + 1, "Gradient contains Inf"
                    )
                elif grad_norm.isnan():
                    logger.warning(
                        "Epoch", epoch, "Step", i + 1, "Gradient contains NaN"
                    )
                else:
                    writer.add_scalar(
                        "train/gradient_norm", grad_norm.item() / ws, global_step
                    )
                    recent_grad_norms.append(grad_norm.item() / ws)

        if dist.is_initialized():
            dist.barrier()

        if not grad_has_inf_nan:
            optimizer.step()
            num_batch_updates += 1
        if dist.is_initialized():
            dist.barrier()

        step_time += time() - start
        start = time()

        if dist.is_initialized():
            dist.barrier()
        losses["all_loss"] = (1, loss.detach())
        for key in losses:
            val = losses[key][1]
            key = f"train_{key}" if "/" in key else f"train/loss/{key}"
            write_scalar_ddp(writer, key, val, global_step, rank, ws, use_ddp)

        if dist.is_initialized():
            dist.barrier()
        for key in stats:
            val = stats[key]
            key = f"train_{key}" if "/" in key else f"train/metric/{key}"
            write_scalar_ddp(writer, key, val, global_step, rank, ws, use_ddp)

        if rank == 0 and writer is not None:
            writer.add_scalar("train/epoch", epoch, global_step)

        log_time += time() - start
        start = time()

        if dist.is_initialized():
            dist.barrier()

        if (i + 1) % cfg.train.print_freq == 0:
            data_time = data_time.item() / ws / cfg.train.print_freq
            step_time = step_time.item() / ws / cfg.train.print_freq
            log_time = log_time.item() / ws / cfg.train.print_freq
            if rank == 0:
                avg_grad_norm_str = ""
                if recent_grad_norms:
                    avg_gn = sum(recent_grad_norms) / len(recent_grad_norms)
                    avg_grad_norm_str = f"  grad_norm {avg_gn:.4f}"
                    recent_grad_norms.clear()
                logger.info(
                    f"Train Epoch {epoch:02d}  {(i+1):04d}/{num_steps:04d}  "
                    f"Data time {data_time:.4f}  Forward time {step_time:.4f}"
                    f"  Logging time {log_time:.4f} Loss {loss.detach():.4f}"
                    f"{avg_grad_norm_str}"
                )
                if writer is not None:
                    writer.add_scalar("timing/data_time", data_time, global_step)
                    writer.add_scalar("timing/forward_time", step_time, global_step)
                    writer.add_scalar("timing/log_time", log_time, global_step)
                    writer.add_scalar(
                        "timing/loss_per_step", loss.detach().item(), global_step
                    )
            data_time = torch.tensor(0.0, device=device_rank)
            step_time = torch.tensor(0.0, device=device_rank)
            log_time = torch.tensor(0.0, device=device_rank)

    return global_step


def eval_one_epoch(loader, model, writer, epoch, global_step, cfg):
    global handler_called
    rank = 0
    ws = 1
    use_ddp = dist.is_available() and cfg.train.num_gpus > 1
    if use_ddp:
        rank = dist.get_rank()
        ws = dist.get_world_size()

    # Derive local CUDA device (matches torch.cuda.set_device in train())
    slurm_localid = os.environ.get("SLURM_LOCALID")
    if slurm_localid is not None:
        device_rank = int(slurm_localid)
    else:
        device_rank = rank

    num_steps = len(loader)
    # num_plots = num_steps // cfg.train.plot_freq
    # plot_ids = torch.randperm(num_steps)[:num_plots]
    data_time = torch.tensor(0.0, device=device_rank)
    step_time = torch.tensor(0.0, device=device_rank)
    log_time = torch.tensor(0.0, device=device_rank)

    total = {}
    loss_dict_epoch, stats_epoch, stats_recon_epoch = {}, {}, {}
    start = time()

    for i, data in enumerate(loader):

        if handler_called:
            logger.info("Terminating training, sayonara!")
            sys.exit(0)

        if data is None:
            continue

        to_gpu(data)
        data_time += time() - start
        start = time()

        with torch.no_grad():
            outputs, losses, stats = model(data, cfg.train)

            if cfg.train.model_name == "diffusion":
                _, _, stats_recon = model(data, eval=True)

            loss = sum([w * v for w, v in losses.values()])
            losses["all_loss"] = (1, loss.detach())
        step_time += time() - start
        start = time()

        for key in losses:
            add_to_dict(loss_dict_epoch, key, losses[key][1])

        for key in stats:
            add_to_dict(stats_epoch, key, stats[key])

        if cfg.train.model_name == "diffusion":
            for key in stats_recon:
                add_to_dict(stats_recon_epoch, key, stats_recon[key])

        log_time += time() - start
        start = time()

        if (i + 1) % cfg.train.print_freq == 0:
            if rank == 0:
                data_time = data_time.item() / ws / cfg.train.print_freq
                step_time = step_time.item() / ws / cfg.train.print_freq
                log_time = log_time.item() / ws / cfg.train.print_freq
                logger.info(
                    f"Valid Epoch {epoch:02d}  {(i+1):04d}/{num_steps:04d}  "
                    f"Data time {data_time:.4f}  Forward time {step_time:.4f}"
                    f"  Logging time {log_time:.4f}"
                )
            data_time = torch.tensor(0.0, device=device_rank)
            step_time = torch.tensor(0.0, device=device_rank)
            log_time = torch.tensor(0.0, device=device_rank)

    total["steps"] = torch.tensor(i + 1, device=device_rank)
    if use_ddp:
        for key in total:
            dist.reduce(total[key], 0)

    write_scalar_ddp(
        writer, f"valid/epoch", epoch, global_step, rank, total["steps"], False
    )

    for key, val in loss_dict_epoch.items():
        write_scalar_ddp(
            writer, f"valid/loss/{key}", val, global_step, rank, total["steps"], use_ddp
        )

    for key, val in stats_epoch.items():
        write_scalar_ddp(
            writer,
            f"valid/metric/noise/{key}",
            val,
            global_step,
            rank,
            total["steps"],
            use_ddp,
        )

    if cfg.train.model_name == "diffusion":
        for key, val in stats_recon_epoch.items():
            write_scalar_ddp(
                writer,
                f"valid/metric/reconstruction/{key}",
                val,
                global_step,
                rank,
                total["steps"],
                use_ddp,
            )


def init_seeds(seed):
    # refer to https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(mode=True, warn_only=True)


def train(rank, use_cache, cfg, shared_caches=None):

    if cfg.data.random_seed != -1:
        seed = cfg.data.random_seed
        logger.info(f"Setting seed to {seed}")
        init_seeds(seed)

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    use_ddp = dist.is_available() and cfg.train.num_gpus > 1

    # Check if running under SLURM with multiple tasks (true multi-node DDP).
    # SLURM_PROCID is set even for single-task srun jobs, so we also check
    # SLURM_NTASKS > 1 to distinguish multi-node (srun launches N processes)
    # from single-node (mp.spawn manages processes internally).
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_localid = os.environ.get("SLURM_LOCALID")
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    is_slurm_multiprocess = slurm_procid is not None and slurm_ntasks > 1

    if use_ddp:
        if is_slurm_multiprocess:
            # Multi-node: use environment variables set by SLURM
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", str(cfg.train.port))
            world_size = int(os.environ.get("WORLD_SIZE", cfg.train.num_gpus))
            rank = int(slurm_procid)  # Override rank from SLURM
            local_rank = (
                int(slurm_localid)
                if slurm_localid is not None
                else rank % cfg.train.num_gpus
            )

            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            logger.info(
                f"SLURM multi-node setup: rank={rank}, local_rank={local_rank}, "
                f"world_size={world_size}, master_addr={master_addr}, master_port={master_port}"
            )
        else:
            # Single-node: use localhost and config (ORIGINAL BEHAVIOR)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(cfg.train.port)
            world_size = cfg.train.num_gpus
            # rank comes from function parameter (mp.spawn)

        os.environ["NCCL_BLOCKING_WAIT"] = "0"
        # Process-group timeout: env-overridable (default 30 min, the
        # PyTorch default) so we can lower it during diagnostics. Was
        # previously hardcoded to 7,200,000 s (~83 days), which made
        # TORCH_NCCL_ASYNC_ERROR_HANDLING's watchdog effectively useless
        # — multinode hangs would never trip it.
        _pg_timeout_s = int(os.environ.get("GRASPGENX_DDP_TIMEOUT_SEC", "1800"))
        dist.init_process_group(
            "nccl",
            timeout=timedelta(seconds=_pg_timeout_s),
            rank=rank,
            world_size=world_size,
        )

        # Set CUDA device: use local_rank for multi-node, rank for single-node
        if is_slurm_multiprocess and slurm_localid is not None:
            device_rank = int(slurm_localid)
        else:
            device_rank = rank  # Single-node: rank parameter from mp.spawn
        torch.cuda.set_device(device_rank)
    else:
        # Single-GPU: this process uses device 0
        device_rank = rank

    # Unpack shared caches (None when single-GPU / debug / cache_mode)
    train_shared = shared_caches[0] if shared_caches is not None else None
    valid_shared = shared_caches[1] if shared_caches is not None else None

    from omegaconf import OmegaConf

    # Inject local rank for shard loading mode (only local_rank 0 copies files)
    loading_mode = cfg.data.get("loading_mode", "preload")
    if loading_mode in ("shard", "shard-gripper"):
        if is_slurm_multiprocess and slurm_localid is not None:
            local_rank = int(slurm_localid)
        else:
            local_rank = rank  # single-node: rank == local rank
        OmegaConf.update(cfg, "data.shard_local_rank", local_rank, force_add=True)

    train_sampler, train_loader = get_xgrasp_data_loader(
        cfg.train,
        cfg.data,
        cfg.train.train_obj_split,
        cfg.train.train_gripper_split,
        cfg.train.train_grasp_split,
        use_ddp,
        use_cache,
        training=True,
        shared_cache=train_shared,
    )

    # Parallel cache mode: only cache the train split for one gripper.
    # Skip validation — it should be cached separately (serial --cache-only
    # or its own parallel run with the valid gripper split).
    if cfg.run == "cache_mode" and cfg.data.get("single_gripper", None) is not None:
        logger.info(
            f"Parallel cache mode for gripper '{cfg.data.single_gripper}': "
            f"train cache saved. Skipping validation. Exiting early."
        )
        return

    valid_sampler, valid_loader = get_xgrasp_data_loader(
        cfg.train,
        cfg.data,
        cfg.train.valid_obj_split,
        cfg.train.valid_gripper_split,
        cfg.train.valid_grasp_split,
        use_ddp,
        use_cache,
        training=False,
        shared_cache=valid_shared,
    )

    # In cache_mode (serial, no single_gripper), both train and valid
    # caches have been generated. Exit early — no model/GPU needed.
    if cfg.run == "cache_mode":
        logger.info("Cache mode: data loaders built and cache saved. Exiting early.")
        return

    if cfg.train.model_name == "diffusion":
        model = GraspGenGenerator.from_config(cfg.diffusion).to(device_rank)
    elif cfg.train.model_name == "discriminator":
        model = GraspGenDiscriminator.from_config(cfg.discriminator).to(device_rank)
    else:
        raise NotImplementedError
    optimizer = build_optimizer(cfg, model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in {cfg.train.model_name} model: {total_params}")

    init_epoch = 0
    init_batch_idx = 0

    logger.info(f"Attempting to load checkpoint from {cfg.train.checkpoint}")
    try:
        ckpt_loaded = False
        if cfg.train.checkpoint is not None:
            if os.path.exists(cfg.train.checkpoint):
                ckpt = torch.load(cfg.train.checkpoint, map_location="cpu")
                init_epoch = ckpt["epoch"]
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                logger.info(f"Loading from checkpoint {cfg.train.checkpoint}")
                init_batch_idx = ckpt["batch_idx"] if "batch_idx" in ckpt else 0
                ckpt_loaded = True
            else:
                logger.warning(f"Checkpoint file not found {cfg.train.checkpoint}")

        # If last.pth was missing, try fallback epoch_*.pth files
        if not ckpt_loaded and cfg.train.checkpoint is not None:
            import glob

            ckpt_dir = cfg.train.log_dir.rstrip("/")
            ckpt_list = [f for f in glob.glob(ckpt_dir + "/epoch_*.pth")]
            if len(ckpt_list) > 0:
                highest_ckpt_idx = sorted(
                    [
                        int(os.path.basename(f).split("epoch_")[1].split(".pth")[0])
                        for f in ckpt_list
                    ]
                )[-1]
                ckpt_file = os.path.join(ckpt_dir, f"epoch_{highest_ckpt_idx}.pth")
                logger.info(f"Falling back to latest epoch checkpoint: {ckpt_file}")
                ckpt = torch.load(ckpt_file, map_location="cpu")
                init_epoch = ckpt["epoch"]
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                init_batch_idx = ckpt["batch_idx"] if "batch_idx" in ckpt else 0
                ckpt_loaded = True
            else:
                logger.warning(
                    "No fallback epoch_*.pth checkpoints found. Starting from scratch."
                )

    except (RuntimeError, EOFError) as e:
        logger.error(e)
        logger.error("Checkpoint is most likely corrupted")

        ckpt_loaded = False
        ckpt_dir = cfg.train.log_dir.rstrip("/")

        # 1) Try last.pth.tmp if it exists (incomplete atomic save)
        tmp_ckpt = os.path.join(ckpt_dir, "last.pth.tmp")
        if os.path.exists(tmp_ckpt):
            logger.info(f"Found {tmp_ckpt}, attempting to recover from it")
            try:
                ckpt = torch.load(tmp_ckpt, map_location="cpu")
                os.replace(tmp_ckpt, os.path.join(ckpt_dir, "last.pth"))
                init_epoch = ckpt["epoch"]
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                init_batch_idx = ckpt["batch_idx"] if "batch_idx" in ckpt else 0
                logger.info(f"Successfully recovered from {tmp_ckpt}")
                ckpt_loaded = True
            except (RuntimeError, EOFError) as e2:
                logger.error(f"last.pth.tmp is also corrupted: {e2}")

        # 2) Fall back to latest epoch_*.pth
        if not ckpt_loaded:
            import glob

            ckpt_list = [f for f in glob.glob(ckpt_dir + "/epoch_*.pth")]
            if len(ckpt_list) > 0:
                highest_ckpt_idx = sorted(
                    [
                        int(os.path.basename(f).split("epoch_")[1].split(".pth")[0])
                        for f in ckpt_list
                    ]
                )[-1]
                ckpt_file = os.path.join(ckpt_dir, f"epoch_{highest_ckpt_idx}.pth")
                logger.info(f"Falling back to latest epoch checkpoint: {ckpt_file}")
                ckpt = torch.load(ckpt_file, map_location="cpu")
                init_epoch = ckpt["epoch"]
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                init_batch_idx = ckpt["batch_idx"] if "batch_idx" in ckpt else 0
            else:
                logger.warning("No fallback checkpoints found. Starting from scratch.")

    batch_idx = init_batch_idx

    if use_ddp:
        # https://github.com/Lightning-AI/lightning/issues/6789
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device_rank], output_device=device_rank)

    clip_grad = None
    if cfg.optimizer.grad_clip > 0:
        clip_grad = partial(clip_grad_norm_, max_norm=cfg.optimizer.grad_clip)

    writer = None
    if rank == 0:
        writer = SummaryWriter(cfg.train.log_dir)
    if rank == 0:
        # Save training configuration to YAML file
        config_save_path = os.path.join(cfg.train.log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            OmegaConf.save(cfg, f)
        logger.info(f"Saved training configuration to {config_save_path}")

    global_step = init_epoch * len(train_loader) + batch_idx
    start = time()

    logger.info(
        f"Training starting at epoch {init_epoch+1} and batch index {batch_idx}"
    )

    for epoch in range(init_epoch, cfg.train.num_epochs):

        # ── Checkpoint C: inside child, before first epoch ──
        if epoch == init_epoch and rank == 0:
            log_all_resources(
                logger, tag=f"rank{rank}_pre_epoch{epoch}", include_gpu=True
            )

        model.train()

        if use_ddp:
            train_sampler.set_epoch(epoch)

        global_step = train_one_epoch(
            train_loader,
            model,
            optimizer,
            clip_grad,
            writer,
            epoch + 1,
            global_step,
            cfg,
            batch_idx,
            rank,
        )

        # ── Checkpoint D: inside child, after first epoch ──
        if epoch == init_epoch and rank == 0:
            log_all_resources(
                logger, tag=f"rank{rank}_post_epoch{epoch}", include_gpu=True
            )

        batch_idx = 0  # Reset to 0 after first (and every) epoch

        if (epoch + 1) % cfg.train.save_freq == 0 and rank == 0:
            save_model(epoch + 1, model, optimizer, cfg.train.log_dir, use_ddp)
            save_model(
                epoch + 1, model, optimizer, cfg.train.log_dir, use_ddp, name="last"
            )
        if (epoch + 1) % cfg.train.eval_freq == 0 or cfg.run == "cache_mode":
            model.eval()
            if use_ddp:
                valid_sampler.set_epoch(epoch)
            eval_one_epoch(valid_loader, model, writer, epoch + 1, global_step, cfg)

        if writer is not None:
            writer.flush()

        if cfg.run == "cache_mode":
            logger.info(
                "Prefiltering mode. Exiting Train script since we iterated one pass over the dataset"
            )
            break

    total_time = torch.tensor(time() - start).to(device_rank)
    if use_ddp:
        dist.reduce(total_time, 0)
    total_time = total_time.item() / cfg.train.num_gpus
    if rank == 0:
        try:
            logger.info("Total training time", timedelta(seconds=total_time))
        except Exception as e:
            logger.error(f"Error logging total training time: {e}")
            pass


@hydra.main(config_path=".", config_name="config_xgrasp", version_base="1.3")
def main(cfg: DictConfig) -> None:

    # ── Checkpoint A: script startup (before any data loading) ──
    log_all_resources(logger, tag="startup", include_gpu=False)

    if cfg.run == "cache_mode":

        cache_dir = get_cache_path(cfg.data.cache_dir, cfg.data.cache_name)

        logger.info("Running in cache mode")
        os.makedirs(cache_dir, exist_ok=True)
        cfg.train.num_gpus = 0
        cfg.train.num_workers = 0
        train(0, False, cfg)
        assert os.path.exists(cache_dir)

    elif cfg.run == "train_mode":

        # Detect number of available GPUs and set cfg.train.num_gpus
        num_gpus_available = torch.cuda.device_count()
        # assert cfg.train.num_gpus <= num_gpus_available
        logger.info(
            f"Detected {num_gpus_available} GPU(s) available, setting cfg.train.num_gpus = {cfg.train.num_gpus}"
        )

        loading_mode = cfg.data.get("loading_mode", "preload")
        logger.info(f"Data loading mode: {loading_mode}")

        # Check if srun launched multiple processes (true multi-node DDP).
        # SLURM_PROCID is always set by srun, even with ntasks=1, so we
        # also require SLURM_NTASKS > 1 to enter the multi-process path.
        slurm_procid = os.environ.get("SLURM_PROCID")
        slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        is_slurm_multiprocess = slurm_procid is not None and slurm_ntasks > 1

        if is_slurm_multiprocess:
            # Multi-node: srun launches one process per GPU, each calls train() directly
            logger.info(
                f"Detected SLURM multi-process environment (ntasks={slurm_ntasks}) - using multi-node DDP launch"
            )
            rank = int(slurm_procid)
            # For multi-node, we can't pre-load caches in parent (no fork)
            # Use lazy loading or pre-load on each node (less efficient but works)
            if loading_mode == "lazy":
                train(rank, True, cfg, None)
            else:
                # Note: In multi-node, each process loads its own cache
                # This is less memory-efficient than single-node COW sharing
                # but necessary since we can't fork across nodes
                logger.warning(
                    "Multi-node with preload mode: each process will load cache independently"
                )
                train(rank, True, cfg, None)
        elif cfg.train.debug:
            train(0, True, cfg)
        elif loading_mode in ("lazy", "shard", "shard-gripper"):
            # Lazy/shard path: each worker reads from H5 on demand — no
            # bulk RAM load, no gc.freeze, no fork requirement.
            # Shard modes progressively copy shards in the background;
            # merged H5 file is not needed on local disk.
            logger.info(
                f"Using {loading_mode} data loading mode — skipping pre-load into RAM"
            )
            mp.spawn(
                train, args=(True, cfg, None), nprocs=cfg.train.num_gpus, join=True
            )
        else:
            # Pre-load dataset caches once in the main process.
            # After fork, all workers share the same physical pages via
            # kernel COW.  gc.freeze() prevents the cyclic GC from
            # scanning (and thereby COW-dirtying) those pages.
            shared_caches = None
            if cfg.train.num_gpus > 1:
                assert (
                    not torch.cuda.is_initialized()
                ), "CUDA must not be initialized before fork-based mp.spawn"
                shared_caches = _pre_load_shared_caches(cfg)

                # ── Checkpoint B: after cache load (before fork) ──
                log_all_resources(logger, tag="after_cache_load", include_gpu=False)

                # Freeze the cyclic GC so it won't walk (and thereby
                # COW-dirty) the remaining small Python wrapper objects
                # (dicts, tuples, etc.) after fork.
                gc.disable()
                gc.freeze()

            mp.spawn(
                train,
                args=(True, cfg, shared_caches),
                nprocs=cfg.train.num_gpus,
                join=True,
                start_method="fork",
            )

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
