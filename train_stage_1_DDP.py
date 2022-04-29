import os
from tqdm import tqdm
from render_func import *
import time
import numpy as np
from mvsd_dataset import Openrooms_org
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
import random
import sys
from cfgnode import CfgNode
import datetime
from models import *
from utils import *

dataRoot = '/home/happily/Data/OpenRooms_org/'
outputRoot = '/home/happily/Data/output/MVSD/stage1'

if not torch.cuda.is_available():
    assert 'gpu isnt ready!'

# debug = True
debug = False
phase = 'TRAIN'
train_type = 'normal'

def main():
    gpus = '0'
    # gpus = '0,1,2,3,4,5,6,7'
    num_gpu = len(gpus.split(','))
    config = 'stage1_1.yml'
    id = '3456'

    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)

    config = config.replace('.yml', '')
    experiment = f'{outputRoot}/{config}'
    os.makedirs(experiment, exist_ok=True)

    with open(os.path.join(experiment, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    if cfg.distributed:
        torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, cfg, experiment, id))
    else:
        train(0, 1, cfg, experiment, id)


def train(gpu, ngpus_per_node, cfg, experiment, id):
    scaler = torch.cuda.amp.GradScaler()
    enable_autocast = cfg.autocast

    experiment_name = experiment.split('/')[-1]

    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:' + id, world_size=ngpus_per_node, rank=gpu)
    torch.cuda.set_device(gpu)

    # create training dataset
    ####################################
    pinned = True
    train_dataset = Openrooms_org(dataRoot, phase, debug, cfg.env_height, cfg.env_row)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                                                   pin_memory=pinned, sampler=train_sampler)

    val_dataset = Openrooms_org(dataRoot, 'TEST', debug, cfg.env_height, cfg.env_row)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    curr_model = SVNModel(cfg, gpu, experiment, phase=phase)

    scalars_to_log = {}
    global_step = curr_model.start_step + 1
    max_iterations = len(train_loader) * cfg.nepoch
    print(f'max iteration : {max_iterations}')

    epoch = 0

    writer = None
    if gpu == 0:
        writer = SummaryWriter(experiment)
        print('saving tensorboard files to {}'.format(experiment))

    # iterations = range(start_step, iterations)
    # for global_step in tqdm(iterations):
    start_time = time.time()
    while global_step < max_iterations:
        np.random.seed()
        for train_data in train_loader:
            if cfg.distributed:
                train_sampler.set_epoch(epoch)
            rgbd = train_data['rgbd'].cuda(gpu, non_blocking=pinned)
            seg = train_data['seg'].cuda(gpu, non_blocking=pinned)
            normal_gt = train_data['normal'].cuda(gpu, non_blocking=pinned)
            light = train_data['light'].cuda(gpu, non_blocking=pinned)
            segAll = seg[:, 1:, ...]
            pixelAllNum = torch.sum(segAll).cpu().data.item() + 1.0
            normal_pred = curr_model.normal_net(rgbd)

            if cfg.normal_loss == 'L2':
                normal_err = torch.sum((normal_pred - normal_gt) * (normal_pred - normal_gt) * segAll) / pixelAllNum / 3.0
            elif cfg.normal_loss == 'ang':
                normal_err = torch.sum(torch.arccos(torch.sum(normal_pred * normal_gt, dim=1)) * segAll) / pixelAllNum
            elif cfg.normal_loss == 'both':
                normal_err = torch.sum((normal_pred - normal_gt) * (normal_pred - normal_gt) * segAll) / pixelAllNum / 3.0
                normal_err += torch.sum(torch.arccos(torch.sum(normal_pred * normal_gt, dim=1)) * segAll) / pixelAllNum
            else:
                raise Exception('normal err type error')

            total_err = normal_err
            # compute loss
            curr_model.optimizer.zero_grad()
            total_err.backward()
            scalars_to_log['normal_err'] = normal_err.item()
            scalars_to_log['total_err'] = total_err.item()
            curr_model.optimizer.step()
            curr_model.scheduler.step()

            scalars_to_log['lr'] = curr_model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            # dt = time.time() - time0

            # Rest is logging
            if gpu == 0:
                for k in scalars_to_log.keys():
                    # logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                    writer.add_scalar('train/' + k, scalars_to_log[k], global_step)
                if global_step % cfg.i_print == 1:
                    # logstr = '{} Epoch: {}  step: {} '.format(experiment_name, epoch, global_step)
                    elapsed_sec = time.time() - start_time
                    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
                    remain_sec = elapsed_sec * max_iterations / global_step
                    remain_time = str(datetime.timedelta(seconds=remain_sec)).split('.')[0]
                    prog = global_step / max_iterations * 100

                    print(f'P{experiment_name} epoch : {epoch} step: {global_step} / {max_iterations}, {prog:.1f} elapsed:{elapsed_time}, remain:{remain_time}')
                    # print('each iter time {:.05f} seconds'.format(dt))

                if global_step % cfg.i_weights == 1:
                    fpath = os.path.join(experiment, 'model_{:06d}.pth'.format(global_step))
                    curr_model.save_model(fpath)

                if global_step % cfg.i_img == 1:
                    print('Logging a random validation view...')
                    val_data = next(val_loader_iterator)
                    log_view_to_tb(writer, global_step, cfg, gpu, curr_model, val_data, prefix='val/')
                    torch.cuda.empty_cache()
            global_step += 1
        epoch += 1

    if gpu == 0:
        fpath = os.path.join(experiment, 'model_latest.pth')
        curr_model.save_model(fpath)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
