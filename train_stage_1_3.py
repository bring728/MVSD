import os
from tqdm import tqdm
from render_func import *
from record import *
import time
import numpy as np
from mvsd_dataset import Openrooms_org
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
import random
import sys
from cfgnode import CfgNode
from models import *
from utils import *

dataRoot = '/media/vig-titan2/mybookduo1/OpenRooms_org'
outputRoot = '/home/vig-titan2/Data/MVSD_output/stage1'

if not torch.cuda.is_available():
    assert 'gpu isnt ready!'

# debug = True
debug = False
phase = 'TRAIN'


def train(gpu, config):
    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    cfg.distributed = False
    model_type = cfg.model_type

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config.replace('.yml', '')
    experiment = f'{outputRoot}/{config}'
    os.makedirs(experiment, exist_ok=True)

    with open(os.path.join(experiment, "config.yml"), "w") as f:
        f.write(cfg.dump())

    scaler = torch.cuda.amp.GradScaler()
    enable_autocast = cfg.autocast

    ckpt_prefix = os.path.join(experiment, f'model_{model_type}')
    exp_name = experiment.split('/')[-1]

    curr_model = SVNDModel(cfg, gpu, experiment, phase=phase)
    sg2env = SG2env(cfg.SGNum, envWidth=cfg.env_height * 2, envHeight=cfg.env_height, gpu=gpu)

    pinned = True
    train_dataset = Openrooms_org(dataRoot, phase, debug, cfg.env_height, cfg.env_width, cfg.env_rows, cfg.env_cols)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.num_workers, pin_memory=pinned)

    val_dataset = Openrooms_org(dataRoot, 'TEST', debug, cfg.env_height, cfg.env_width, cfg.env_rows, cfg.env_cols)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batchsize, shuffle=False)

    scalars_to_log = {}
    global_step = len(train_loader) * curr_model.start_epoch + 1
    max_iterations = len(train_loader) * cfg.nepoch

    epoch = curr_model.start_epoch
    writer = SummaryWriter(experiment)
    print('saving tensorboard files to {}'.format(experiment))

    start_time = time.time()
    while global_step < max_iterations:
        for train_data in train_loader:
            with autocast(enabled=enable_autocast):
                rgbd = train_data['rgbd'].cuda(gpu, non_blocking=pinned)
                normal_gt = train_data['normal'].cuda(gpu, non_blocking=pinned)
                envmaps_gt = train_data['envmaps'].cuda(gpu, non_blocking=pinned)
                envmapsIndBatch = train_data['envmapsInd'].cuda(gpu, non_blocking=pinned)
                seg = train_data['seg']
                segAll = seg[:, 1:, ...].cuda(gpu, non_blocking=pinned)
                segBRDF = seg[:, :1, ...].cuda(gpu, non_blocking=pinned)

                normal_err = 0.0
                env_err = 0.0

                if cfg.model_type == 'normal':
                    normal_pred = curr_model.normal_net(rgbd)
                    if cfg.normal_loss == 'L2':
                        normal_err = img2mse(normal_pred, normal_gt, segAll)
                    elif cfg.normal_loss == 'ang':
                        normal_err = img2angerr(normal_pred, normal_gt, segAll)
                    elif cfg.normal_loss == 'both':
                        normal_err = img2mse(normal_pred, normal_gt, segAll)
                        normal_err += img2angerr(normal_pred, normal_gt, segAll)
                    else:
                        raise Exception('normal err type error')
                    scalars_to_log['normal_err'] = normal_err.item()

                elif cfg.model_type == 'light':
                    with torch.no_grad():
                        normal_pred = curr_model.normal_net(rgbd)
                    rgbdn = torch.cat([rgbd, normal_pred], dim=1)
                    axis, sharpness, intensity = curr_model.DL_net(rgbdn)
                    segBRDF = F.adaptive_avg_pool2d(segBRDF, (cfg.env_rows, cfg.env_cols))
                    notDarkEnv = (torch.mean(torch.mean(torch.mean(envmaps_gt, 4), 4), 1, True) > 0.001).float()
                    segEnvBatch = (segBRDF * envmapsIndBatch.expand_as(segBRDF)).unsqueeze(-1).unsqueeze(-1)
                    segEnvBatch = (segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)).expand_as(envmaps_gt)

                    envmap_pred = sg2env.forward(axis, sharpness, intensity)
                    if cfg.scale_inv:
                        envmap_pred = LSregress(envmap_pred.detach() * segEnvBatch, envmaps_gt * segEnvBatch, envmap_pred)

                    env_err = img2log_mse(envmap_pred, envmaps_gt, segEnvBatch)
                    scalars_to_log['env_err'] = env_err.item()

                total_err = normal_err + env_err

            curr_model.optimizer.zero_grad()
            scaler.scale(total_err).backward()
            scaler.step(curr_model.optimizer)
            scaler.update()

            curr_model.scheduler.step()
            scalars_to_log['total_err'] = total_err.item()
            scalars_to_log['lr'] = curr_model.scheduler.get_last_lr()[0]

            for k in scalars_to_log.keys():
                writer.add_scalar('train/' + k, scalars_to_log[k], global_step)
            if global_step % cfg.i_print == 0:
                print_state(exp_name, start_time, max_iterations, global_step, epoch, gpu)
            if global_step % cfg.i_img == 0:
                if cfg.model_type == 'normal':
                    record_image(writer, epoch, global_step, (0.5 * (normal_pred[0] + 1)), (0.5 * (normal_gt[0] + 1)), 'normal')
                elif cfg.model_type == 'light':
                    envmap_pred = torch.permute(envmap_pred[0] / torch.max(envmap_pred[0]), (0, 1, 3, 2, 4)).reshape(3, cfg.env_rows * cfg.env_height, cfg.env_cols * cfg.env_width)
                    envmaps_gt = torch.permute(envmaps_gt[0] / torch.max(envmaps_gt[0]), (0, 1, 3, 2, 4)).reshape(3, cfg.env_rows * cfg.env_height, cfg.env_cols * cfg.env_width)
                    record_image(writer, epoch, global_step, envmap_pred, envmaps_gt, 'lighting')
            global_step += 1

        epoch += 1
        fpath = f'{ckpt_prefix}_{epoch:06d}.pth'
        curr_model.save_model(fpath)

        eval_model(writer, curr_model, sg2env, val_loader, gpu, epoch, cfg)
        torch.cuda.empty_cache()

    fpath = f'{ckpt_prefix}_latest.pth'
    curr_model.save_model(fpath)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        gpu = 0
        config = 'stage1_3.yml'
    else:
        gpu = int(sys.argv[1])
        config = sys.argv[2]
    train(gpu, config)
