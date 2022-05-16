from record import *
import time
from tqdm import tqdm
import numpy as np
from mvsd_dataset import *
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
import random
import sys
from cfgnode import CfgNode
from models import *
from utils import *

dataRoot = '/media/vig-titan-103/mybookduo/OpenRooms_FF'
outputRoot = '/home/vig-titan-103/Data/MVSD_output/stage1-2'

torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    assert 'gpu isnt ready!'

def train(gpu, num_gpu, config, debug=False, phase='TRAIN', is_DDP=False):
    model_type = 'light'
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2958', world_size=num_gpu, rank=gpu)
        torch.cuda.set_device(gpu)

    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    record_flag = (is_DDP and gpu == 0) or not is_DDP

    config = config.replace('.yml', '')
    experiment = f'{outputRoot}/{config}'
    os.makedirs(experiment, exist_ok=True)

    with open(os.path.join(experiment, "config.yml"), "w") as f:
        f.write(cfg.dump())

    scaler = torch.cuda.amp.GradScaler()
    enable_autocast = cfg.autocast

    ckpt_prefix = os.path.join(experiment, f'model_normal')
    exp_name = experiment.split('/')[-1]

    curr_model = SVDirectLightModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)
    sg2env = SG2env(cfg.SGNum, envWidth=cfg.env_width, envHeight=cfg.env_height, gpu=gpu)

    if debug:
        batch_per_gpu = 1
        worker_per_gpu = 0
    else:
        batch_per_gpu = int(cfg.batchsize / num_gpu)
        worker_per_gpu = int(cfg.num_workers / num_gpu)

    print('batch_per_gpu', batch_per_gpu, 'worker_per_gpu', worker_per_gpu)
    train_dataset = Openrooms_FF_single(dataRoot, cfg, phase, debug)
    pinned = True
    if is_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, shuffle=False, num_workers=worker_per_gpu,
                                  pin_memory=pinned, sampler=train_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, shuffle=True, num_workers=worker_per_gpu,
                                  pin_memory=pinned, drop_last=True)

    val_dataset = Openrooms_FF_single(dataRoot, cfg, 'TEST', debug)
    val_loader = DataLoader(val_dataset, batch_size=batch_per_gpu, shuffle=False)

    scalars_to_log = {}
    global_step = len(train_loader) * curr_model.start_epoch + 1
    max_iterations = len(train_loader) * cfg.nepoch

    epoch = curr_model.start_epoch
    if record_flag:
        writer = SummaryWriter(experiment)
        print('saving tensorboard files to {}'.format(experiment))

    start_time = time.time()
    while global_step < max_iterations:
        if is_DDP:
            train_sampler.set_epoch(epoch)
        for train_data in tqdm(train_loader):
            with autocast(enabled=enable_autocast):
                train_data = tocuda(train_data, gpu)
                rgbdc = train_data['input']
                segBRDF = train_data['seg'][:, :1, ...]
                envmaps_gt = train_data['envmaps_gt']
                envmapsIndBatch = train_data['envmapsInd']

                with torch.no_grad():
                    normal_pred = curr_model.normal_net(rgbdc)
                rgbdcn = torch.cat([rgbdc, 0.5 * (normal_pred + 1)], dim=1)
                axis, sharpness, intensity = curr_model.DL_net(rgbdcn)
                segBRDF = F.adaptive_avg_pool2d(segBRDF, (cfg.env_rows, cfg.env_cols))
                notDarkEnv = (torch.mean(torch.mean(torch.mean(envmaps_gt, 4), 4), 1, True) > 0.001).float()
                segEnvBatch = (segBRDF * envmapsIndBatch.expand_as(segBRDF)).unsqueeze(-1).unsqueeze(-1)
                segEnvBatch = (segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)).expand_as(envmaps_gt)

                envmaps_pred = sg2env.forward(axis, sharpness, intensity)
                envmaps_pred_scaled = LSregress(envmaps_pred.detach() * segEnvBatch, envmaps_gt * segEnvBatch, envmaps_pred)
                env_err = img2log_mse(envmaps_pred_scaled, envmaps_gt, segEnvBatch)

            scalars_to_log['env_err'] = env_err.item()
            curr_model.optimizer.zero_grad()
            scaler.scale(env_err).backward()
            scaler.step(curr_model.optimizer)
            scaler.update()

            curr_model.scheduler.step()
            scalars_to_log['lr'] = curr_model.scheduler.get_last_lr()[0]

            global_step += 1
            if record_flag:
                for k in scalars_to_log.keys():
                    writer.add_scalar('train/' + k, scalars_to_log[k], global_step)
                if global_step % cfg.i_print == 0:
                    print_state(exp_name, start_time, max_iterations, global_step, epoch, gpu)
                if global_step % cfg.i_img == 0:
                    imgs = [env_vis(envmaps_gt[0], cfg.env_height, cfg.env_width, cfg.env_rows, cfg.env_cols),
                            env_vis(envmaps_pred_scaled[0], cfg.env_height, cfg.env_width, cfg.env_rows, cfg.env_cols)]
                    record_images(writer, global_step, imgs, model_type)
                    normal_gt = train_data['normal_gt']
                    depth_gt = train_data['depth_gt'][0]
                    depth_input = train_data['input'][0, 3, ...]
                    imgs = [(0.5 * (normal_gt[0] + 1)), (0.5 * (normal_pred[0] + 1)), depth_gt.expand(normal_gt[0].size()),
                            depth_input.expand(normal_gt[0].size())]
                    record_images(writer, global_step, imgs, model_type + '_input')

        epoch += 1
        if record_flag:
            fpath = f'{ckpt_prefix}_{epoch:06d}.pth'
            curr_model.save_model(fpath)
            eval_model(writer, curr_model, sg2env, val_loader, gpu, epoch, cfg, model_type)
            torch.cuda.empty_cache()

    if record_flag:
        fpath = f'{ckpt_prefix}_latest.pth'
        curr_model.save_model(fpath)
        quality_eval_model(writer, curr_model, sg2env, val_loader, gpu, cfg, model_type)

    if is_DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train(gpu=0, num_gpu=1, debug=False, phase='TRAIN', config='stage1-2_0.yml', is_DDP=False)
