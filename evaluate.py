import os

import torch
from tqdm import tqdm
from CIS.cis_model import CIS_model
from mvsd_dataset import Openrooms_FF
from torch.utils.data import DataLoader
from cfgnode import CfgNode
from utils import *
from forward import model_forward
from models import *
import yaml


a = 'Batch'

outfilename = 'cis_output'
os.makedirs(outfilename, exist_ok=True)
dataRoot = '/home/happily/Data/OpenRooms_FF'
config = 'stage2_0.yml'
experiment = '/home/happily/Data/MVSD_output/stage2/06131105_stage2/'

with open(os.getcwd() + '/config/' + config, "r") as f:
    cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = CfgNode(cfg_dict)

val_dataset = Openrooms_FF(dataRoot, cfg, '2', 'TEST', False)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=3, shuffle=False)
cis_model = CIS_model()

helper_dict = {}
u, v = np.meshgrid(np.arange(cfg.imWidth), np.arange(cfg.imHeight))
u = u.astype(dtype=np.float32) + 0.5  # add half pixel
u_n = 2.0 * u / cfg.imWidth - 1
v = v.astype(dtype=np.float32) + 0.5
v_n = 2.0 * v / cfg.imHeight - 1
pixels = np.stack((u, v, np.ones_like(u), u_n, v_n), axis=-1)
pixels = torch.from_numpy(pixels)
pixels = pixels.to(0)
helper_dict['pixels'] = pixels
mvsd_model = BRDFModel(cfg, 0, experiment, phase='TEST', is_DDP=False)


j = 0
scalars_to_log_all = {}
scalars_to_log = {}
for data in tqdm(val_loader):
    j += 1
    target_rgb = np.transpose(data['rgb'][0, 0].numpy(), [1, 2, 0])
    dict_forward = cis_model.forward(target_rgb)

    val_data = tocuda(data, 0, False)
    _ = model_forward('2', 'TEST', mvsd_model, helper_dict, val_data, cfg, scalars_to_log, False)
    for k in scalars_to_log:
        val_k = k.replace('train', 'val')
        if val_k in scalars_to_log_all:
            scalars_to_log_all[val_k] += scalars_to_log[k]
        else:
            scalars_to_log_all[val_k] = 0.0
