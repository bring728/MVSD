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

    experiment = '/new_disk/happily/Data/MVSD_output/stage1-1/20220517_stage1-1_group_1e-4_64'

    device = torch.device('cuda:{}'.format(gpu))
    curr_model = MonoNormalModel(cfg, gpu, experiment, phase=phase, is_DDP=is_DDP)
    im = loadHdr('/new_disk/happily/Data/OpenRooms_FF/mainDiffLight_xml/scene0151_00/1_im_5.rgbe')
    seg = loadImage('/new_disk/happily/Data/OpenRooms_FF/mainDiffLight_xml/scene0151_00/1_immask_5.png')[0:1, :, :]
    scene_scale = get_hdr_scale(im, seg, 'TEST')
    im = np.clip(im * scene_scale, 0, 1.0)

    depth = loadBinary('/new_disk/happily/Data/OpenRooms_FF/mainDiffLight_xml/scene0151_00/1_depthestnorm_5.dat')
    conf = loadBinary('/new_disk/happily/Data/OpenRooms_FF/mainDiffLight_xml/scene0151_00/1_depthestnorm_5.dat')
    normal_pred = curr_model.normal_net(torch.from_numpy(np.concatenate([im, depth, conf]).astype(np.float32))[None].to(device))
    normal = 0.5 * (normal_pred + 1)
    print()

    # val_dataset = Openrooms_FF_single(dataRoot, cfg, 'TEST', debug)
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    #
    # writer = SummaryWriter(experiment)
    #
    # eval_model(writer, curr_model, None, val_loader, gpu, 31, cfg, 'normal', True)




if __name__ == "__main__":
    train(gpu=0, num_gpu=1, debug=False, phase='TEST', config='stage1-1_0.yml', is_DDP=False)
