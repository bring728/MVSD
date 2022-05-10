from record import *
import time
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
outputRoot = '/home/vig-titan-103/Data/MVSD_output/stage1-1'

if not torch.cuda.is_available():
    assert 'gpu isnt ready!'

#
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def train(gpu, num_gpu, config, debug=False, phase='TRAIN', is_DDP=False):
    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config.replace('.yml', '')
    experiment = f'{outputRoot}/{config}'

    curr_model = SVNormalModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)

    val_dataset = Openrooms_FF_single(dataRoot, cfg, 'TEST', debug)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    writer = SummaryWriter(experiment)

    eval_model(writer, curr_model, None, val_loader, gpu, 31, cfg, 'normal', True)




if __name__ == "__main__":
    train(gpu=0, num_gpu=1, debug=False, phase='TEST', config='stage1-1_0.yml', is_DDP=False)
