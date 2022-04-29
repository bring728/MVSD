from record import *
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

def train(gpu, config):
    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    cfg.distributed = False

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = config.replace('.yml', '')
    experiment = f'{outputRoot}/{config}'
    os.makedirs(experiment, exist_ok=True)

    with open(os.path.join(experiment, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    val_dataset = Openrooms_org(dataRoot, 'TEST', debug, cfg.env_height, cfg.env_row)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batchsize, shuffle=False, num_workers=6)

    curr_model = SVNModel(cfg, gpu, experiment, phase=phase)
    cpkts = curr_model.ckpts

    writer = SummaryWriter(experiment)
    for ckpt in cpkts:
        curr_model.load_model(ckpt)
        epoch = ckpt[-10:-4]
        eval_model(writer, curr_model, val_loader, gpu, epoch)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        gpu = 0
        config = 'stage1_0.yml'
    else:
        gpu = int(sys.argv[1])
        config = sys.argv[2]
    train(gpu, config)
