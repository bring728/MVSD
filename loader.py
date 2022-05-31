from models import MonoNormalModel, MonoDirectLightModel, SG2env, BRDFModel
import copy
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


def load_id_wandb(config, record_flag, resume, root):
    stage = config.split('stage')[1].split('_')[0]
    if stage == '1-1':
        model_type = 'normal'
    elif stage == '1-2':
        model_type = 'DL'
    elif stage == '2':
        model_type = 'BRDF'
    else:
        raise Exception('stage error.')

    outputRoot = osp.join(root, f'MVSD_output/stage{stage}')
    dataRoot = osp.join(root, 'OpenRooms_FF')
    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    wandb_obj = None
    if resume:
        run_id = sorted(os.listdir(outputRoot))[-1]
        if record_flag:
            wandb_obj = wandb.init(project=f'MVSD-stage{stage}', id=run_id, resume='must')
            # wandb_obj.config.update(cfg)
            cfg_saved = CfgNode(wandb_obj.config._items)
            for k in cfg_saved.keys():
                if k == '_wandb':
                    continue
                if cfg_saved[k] != cfg[k]:
                    print('config does not match.')
                    raise Exception('config does not match.')
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


def load_dataloader(stage, dataRoot, cfg, phase, debug, is_DDP, num_gpu, record_flag):
    val_loader = None
    worker_per_gpu = int(cfg.num_workers / num_gpu)
    if stage.startswith('1'):
        train_dataset = Openrooms_FF_single(dataRoot, cfg, stage, phase, debug)
        if not phase == 'ALL':
            val_dataset = Openrooms_FF_single(dataRoot, cfg, stage, 'TEST', debug)
        batch_per_gpu = int(cfg.batchsize / num_gpu)
    else:
        train_dataset = Openrooms_FF(dataRoot, cfg, stage, phase, debug)
        if not phase == 'ALL':
            val_dataset = Openrooms_FF(dataRoot, cfg, stage, 'TEST', debug)
        batch_per_gpu = 1

    if debug and not phase == 'ALL':
        batch_per_gpu = 1
        worker_per_gpu = 0

    train_sampler = None
    if is_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # is_shuffle = not is_DDP and not debug
    is_shuffle = not is_DDP
    train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, shuffle=is_shuffle, num_workers=worker_per_gpu,
                              pin_memory=cfg.pinned, sampler=train_sampler)
    if not phase == 'ALL':
        val_loader = DataLoader(val_dataset, batch_size=batch_per_gpu, num_workers=worker_per_gpu, shuffle=False)

    if record_flag:
        print(f'create dataset - stage {stage}.')
        print('total number of sample: %d' % train_dataset.count)
        print('batch_per_gpu', batch_per_gpu, 'worker_per_gpu', worker_per_gpu)
    return train_loader, val_loader, train_sampler


def load_model(stage, cfg, gpu, experiment, phase, is_DDP, wandb_obj):
    do_watch = wandb_obj is not None
    helper_dict = {}
    curr_model = None

    if stage == '1-1':
        curr_model = MonoNormalModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)
        if do_watch:
            wandb_obj.watch(curr_model.normal_net)

    elif stage == '1-2':
        helper_dict['sg2env'] = SG2env(cfg.DL.SGNum, envWidth=cfg.DL.env_width, envHeight=cfg.DL.env_height, gpu=gpu)
        curr_model = MonoDirectLightModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)
        if do_watch:
            wandb_obj.watch(curr_model.DL_net)

    elif stage == '2':
        helper_dict['sg2env'] = SG2env(cfg.DL.SGNum, envWidth=cfg.DL.env_width, envHeight=cfg.DL.env_height, gpu=gpu)
        u, v = np.meshgrid(np.arange(cfg.imWidth), np.arange(cfg.imHeight))
        # u = u.reshape(-1).astype(dtype=np.float32) + 0.5  # add half pixel
        # u_n = 2.0 * u / cfg.imWidth - 1
        # v = v.reshape(-1).astype(dtype=np.float32) + 0.5
        # v_n = 2.0 * v / cfg.imHeight - 1
        u = u.astype(dtype=np.float32) + 0.5  # add half pixel
        u_n = 2.0 * u / cfg.imWidth - 1
        v = v.astype(dtype=np.float32) + 0.5
        v_n = 2.0 * v / cfg.imHeight - 1
        pixels = np.stack((u, v, np.ones_like(u), u_n, v_n), axis=-1)
        pixels = torch.from_numpy(pixels)
        pixels = pixels.to(gpu, non_blocking=cfg.pinned)
        helper_dict['pixels'] = pixels
        curr_model = BRDFModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)
        if do_watch:
            wandb_obj.watch([curr_model.feature_net, curr_model.brdf_net, curr_model.brdf_refine_net], log='all')

    return curr_model, helper_dict
