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


def train(gpu, num_gpu, config, debug=False, phase='TRAIN', is_DDP=False):
    model_type = 'light'

    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config.replace('.yml', '')
    experiment = f'{outputRoot}/{config}'
    os.makedirs(experiment, exist_ok=True)

    with open(os.path.join(experiment, "config.yml"), "w") as f:
        f.write(cfg.dump())

    device = torch.device('cuda:{}'.format(gpu))
    root = osp.dirname(osp.dirname(experiment))
    normalnet_path = osp.join(root, 'stage1-1', cfg.normalnet_path)
    normal_net = NormalNet(cfg).to(device)
    normal_ckpt = torch.load(osp.join(normalnet_path, 'model_normal_latest.pth'))
    normal_net.load_state_dict(normal_ckpt['normal_net'])

    DL_net = DirectLightingNet(cfg).to(device)
    optimizer = torch.optim.Adam(DL_net.parameters(), )
    sg2env = SG2env(cfg.SGNum, envWidth=cfg.env_width, envHeight=cfg.env_height, gpu=gpu)

    batch_per_gpu = int(cfg.batchsize / num_gpu)
    worker_per_gpu = int(cfg.num_workers / num_gpu)

    print('batch_per_gpu', batch_per_gpu, 'worker_per_gpu', worker_per_gpu)
    train_dataset = Openrooms_FF_single(dataRoot, cfg, phase, debug)
    train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, shuffle=True, num_workers=worker_per_gpu,
                              pin_memory=True, drop_last=True)

    for train_data in tqdm(train_loader):
        train_data = tocuda(train_data, gpu)
        rgbdc = train_data['input']
        segBRDF = train_data['seg'][:, :1, ...]
        envmaps_gt = train_data['envmaps_gt']
        envmapsIndBatch = train_data['envmapsInd']

        with torch.no_grad():
            normal_pred = normal_net(rgbdc)
        rgbdcn = torch.cat([rgbdc, 0.5 * (normal_pred + 1)], dim=1)
        axis, sharpness, intensity = DL_net(rgbdcn)
        segBRDF = F.adaptive_avg_pool2d(segBRDF, (cfg.env_rows, cfg.env_cols))
        notDarkEnv = (torch.mean(torch.mean(torch.mean(envmaps_gt, 4), 4), 1, True) > 0.001).float()
        segEnvBatch = (segBRDF * envmapsIndBatch.expand_as(segBRDF)).unsqueeze(-1).unsqueeze(-1)
        segEnvBatch = (segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)).expand_as(envmaps_gt)

        envmaps_pred = sg2env.forward(axis, sharpness, intensity)
        envmaps_pred_scaled = LSregress(envmaps_pred.detach() * segEnvBatch, envmaps_gt * segEnvBatch, envmaps_pred)
        env_err = img2log_mse(envmaps_pred_scaled, envmaps_gt, segEnvBatch)

        optimizer.zero_grad()
        env_err.backward()
        optimizer.step()


if __name__ == "__main__":
    train(gpu=0, num_gpu=1, debug=False, phase='TRAIN', config='stage1-2_0.yml', is_DDP=False)
