from models import *
# import pyarrow as pa
# import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader
from mvsd_dataset import Openrooms_FF, Openrooms_FF_single
from datetime import datetime
import os
import os.path as osp
from cfgnode import CfgNode
import yaml
import wandb


def load_id_wandb(config, record_flag, resume, root, id=None):
    stage = config.split('stage')[1].split('_')[0]
    if stage == '1':
        model_type = 'singleview'
    elif stage == '2':
        model_type = 'multiview'
    else:
        raise Exception('stage error.')

    outputRoot = osp.join(root, f'MVSD_output/stage{stage}')
    dataRoot = osp.join(root, 'OpenRooms_FF_320')
    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    wandb_obj = None
    if resume:
        if id is None:
            raise Exception('run_id is None')
        # run_id = sorted(os.listdir(outputRoot))[-1]
        run_id = id
        print('resume: ', run_id)
        if record_flag:
            wandb_obj = wandb.init(project=f'MVSD-stage{stage}', id=run_id, resume=True)
            # wandb_obj.config.update(cfg)
            # cfg_saved = CfgNode(wandb_obj.config._items)
            # for k in cfg_saved.keys():
            #     if k == '_wandb':
            #         continue
            #     if cfg_saved[k] != cfg[k]:
            #         print('config does not match.')
            #         raise Exception('config does not match.')
    else:
        current_time = datetime.now().strftime('%m%d%H%M')
        run_id = f'{current_time}_stage{stage}'
        if record_flag:
            wandb_obj = wandb.init(project=f'MVSD-stage{stage}', id=run_id)
            wandb_obj.config.update(cfg)

    experiment = osp.join(outputRoot, run_id)
    if record_flag:
        os.makedirs(experiment, exist_ok=True)
    return stage, cfg, model_type, run_id, wandb_obj, dataRoot, outputRoot, experiment


def load_dataloader(stage, dataRoot, cfg, debug, is_DDP, num_gpu, record_flag):
    if debug:
        worker_per_gpu = 0
        batch_per_gpu = 4
    else:
        worker_per_gpu = cfg.num_workers
        batch_per_gpu = cfg.batchsize

    if stage == '1':
        train_dataset = Openrooms_FF_single(dataRoot, cfg, stage, 'TRAIN')
        val_dataset = Openrooms_FF_single(dataRoot, cfg, stage, 'TEST')
    else:
        train_dataset = Openrooms_FF(dataRoot, cfg, stage, 'TRAIN', debug)
        val_dataset = Openrooms_FF(dataRoot, cfg, stage, 'TEST', debug)

    train_sampler = None
    if is_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    is_shuffle = not is_DDP
    train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, shuffle=is_shuffle, num_workers=worker_per_gpu,
                              pin_memory=cfg.pinned, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_per_gpu, num_workers=worker_per_gpu, shuffle=False)
    if record_flag:
        print(f'create dataset - stage {stage}, shuffle: {is_shuffle}')
        print('total number of sample: %d' % train_dataset.length)
        print('batch_per_gpu', batch_per_gpu, 'worker_per_gpu', worker_per_gpu)
    return train_loader, val_loader, train_sampler


def load_model(stage, cfg, gpu, experiment, phase, is_DDP, wandb_obj):
    do_watch = wandb_obj is not None
    helper_dict = {}
    curr_model = None

    if stage == '1':
        curr_model = SingleViewModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)
        if do_watch:
            watch_model = []
            if cfg.mode == 'normal' or cfg.mode == 'finetune':
                watch_model.append(curr_model.normal_net)
            if cfg.mode == 'DL' or cfg.mode == 'finetune':
                helper_dict['sg2env'] = SG2env(cfg.DL.SGNum, envWidth=cfg.DL.env_width, envHeight=cfg.DL.env_height, gpu=gpu)
                watch_model.append(curr_model.DL_net)
            wandb_obj.watch(watch_model, log='all')

    elif stage == '2':
        helper_dict['sg2env'] = SG2env(cfg.DL.SGNum, envWidth=cfg.DL.env_width, envHeight=cfg.DL.env_height, gpu=gpu)
        u, v = np.meshgrid(np.arange(cfg.imWidth), np.arange(cfg.imHeight), indexing='xy')
        # u = u.reshape(-1).astype(dtype=np.float32) + 0.5  # add half pixel
        # u_n = 2.0 * u / cfg.imWidth - 1
        # v = v.reshape(-1).astype(dtype=np.float32) + 0.5
        # v_n = 2.0 * v / cfg.imHeight - 1
        u = u.astype(dtype=np.float32) + 0.5  # add half pixel
        u_n = 2.0 * u / cfg.imWidth - 1
        v = v.astype(dtype=np.float32) + 0.5
        v_n = 2.0 * v / cfg.imHeight - 1
        pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
        pixels = torch.from_numpy(pixels)
        pixels = pixels.to(gpu, non_blocking=cfg.pinned)[None, :, :, :3, None]
        helper_dict['pixels'] = pixels
        pixels_norm = np.stack((u_n, v_n), axis=-1)
        pixels_norm = torch.from_numpy(pixels_norm)
        pixels_norm = pixels_norm.to(gpu, non_blocking=cfg.pinned)[None].expand([cfg.batchsize, -1, -1, -1])
        helper_dict['pixels_norm'] = pixels_norm
        up = torch.from_numpy(np.array([0, 1.0, 0], dtype=np.float32)).to(gpu, non_blocking=cfg.pinned)
        helper_dict['up'] = up
        curr_model = MultiViewModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)
        if cfg.mode == 'SVL' or cfg.mode == 'finetune':
            x, y, z = np.meshgrid(np.arange(cfg.SVL.vsg_res), np.arange(cfg.SVL.vsg_res), np.arange(cfg.SVL.vsg_res), indexing='xy')
            x = x.astype(dtype=np.float32) + 0.5  # add half pixel
            x = 2.0 * x / cfg.SVL.vsg_res - 1
            y = y.astype(dtype=np.float32) + 0.5
            y = 2.0 * y / cfg.SVL.vsg_res - 1
            z = z.astype(dtype=np.float32) + 0.5
            z = 2.0 * z / cfg.SVL.vsg_res - 1
            voxel_grid_full = np.stack([x, y, z], axis=-1)

            voxel_grid_front = torch.from_numpy(voxel_grid_full[:, :, cfg.SVL.vsg_res // 2:, :])
            voxel_grid_front = voxel_grid_front.to(gpu, non_blocking=cfg.pinned)[None].expand([cfg.batchsize, -1, -1, -1, -1])
            helper_dict['voxel_grid_front'] = voxel_grid_front
            voxel_grid_back = torch.from_numpy(voxel_grid_full[:, :, :cfg.SVL.vsg_res // 2, :])
            voxel_grid_back = voxel_grid_back.to(gpu, non_blocking=cfg.pinned)[None].expand([cfg.batchsize, -1, -1, -1, -1])
            helper_dict['voxel_grid_back'] = voxel_grid_back

            #azimuth, elevation
            Az = ((np.arange(cfg.SVL.env_width) + 0.5) / cfg.SVL.env_width - 0.5) * 2 * np.pi  # -pi ~ pi
            El = ((np.arange(cfg.SVL.env_height) + 0.5) / cfg.SVL.env_height) * np.pi / 2.0  # 0 ~ pi/2
            r = np.arange(cfg.SVL.vsg_res) / cfg.SVL.vsg_res * 2 * (3 ** (1 / 2))
            Az, El, r = np.meshgrid(Az, El, r, indexing='xy')
            Az = Az.reshape(-1, 1)
            El = El.reshape(-1, 1)
            r = r.reshape(-1, 1)
            lx = r * np.sin(El) * np.cos(Az)
            ly = r * np.sin(El) * np.sin(Az)
            lz = r * np.cos(El)
            ls = torch.from_numpy(np.concatenate((lx, ly, lz), axis=-1).astype(np.float32))
            ls = ls.to(gpu, non_blocking=cfg.pinned)[None, None, None].expand([cfg.batchsize, -1, -1, -1, -1])
            helper_dict['ls'] = ls

        if do_watch:
            watch_model = []
            if cfg.mode == 'BRDF' or cfg.mode == 'finetune':
                watch_model += [curr_model.context_net, curr_model.aggregation_net, curr_model.brdf_refine_net]
            if cfg.mode == 'SVL' or cfg.mode == 'finetune':
                watch_model += [curr_model.GL_Net, curr_model.VSG_Net]
            wandb_obj.watch(watch_model, log='all')
    return curr_model, helper_dict
