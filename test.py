import os
from tqdm import tqdm
from render_func import *
import time
from mvsd_dataset import Openrooms_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import random
import sys
from cfgnode import CfgNode
from models import MVSDModel
from utils import *


dataRoot = '/home/happily/Data/OpenRoomsDataset/data/rendering/data_FF_10_640/'
outputRoot = '/home/happily/Data/output/MVSD/'
# dataRoot = 'D:/OpenRoomsDataset/data/rendering/data_FF_10_640/'
# output = 'D:/MVSD_output/mpi-640'


if not torch.cuda.is_available():
    assert 'gpu isnt ready!'

# debug = True
debug = False
phase = 'TEST'

def main():
    if len(sys.argv) < 3:
        gpu = int('0')
        experiment = f'{outputRoot}mpi_dumb_2_1/'
    else:
        gpu = int(sys.argv[1])
        experiment = f'{outputRoot}{sys.argv[2]}/'

    torch.cuda.set_device(gpu)
    with open(f'{experiment}config.yml', "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)

    # current_time = datetime.now()
    out_dir = f'{experiment}test/'
    os.makedirs(out_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    enable_autocast = cfg.autocast

    test_dataset = Openrooms_dataset(dataRoot, num_view_min=cfg.num_view_min, num_view_max=cfg.num_view_max, phase=phase, debug=debug)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # Create IBRNet model
    mvsd_model = MVSDModel(cfg, gpu, experiment, phase=phase)

    scalars_to_log = {}
    u, v = np.meshgrid(np.arange(cfg.imWidth), np.arange(cfg.imHeight))
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5  # add half pixel
    u_n = 2.0 * u / cfg.imWidth - 1
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    v_n = 2.0 * v / cfg.imHeight - 1
    pixels = np.stack((u, v, np.ones_like(u), u_n, v_n), axis=1)
    pixels = torch.from_numpy(pixels).cuda()

    prefix = 'test/'
    epoch = 0

    writer = SummaryWriter(experiment)
    print('saving tensorboard files to {}'.format(experiment))

    psnr_albedo = 0.0
    psnr_normal = 0.0
    psnr_rough = 0.0
    i_image = 10
    for i, test_data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            ret = decompose_single_image(mvsd_model, gpu, cfg.chunk_size, pixels, test_data)

        psnr_albedo += img2psnr(ret['albedo'], ret['albedo_gt'], ret['segBRDF'])
        psnr_normal += img2psnr(ret['normal'], ret['normal_gt'], ret['segAll'])
        psnr_rough += img2psnr(ret['roughness'], ret['roughness_gt'], ret['segBRDF'])

        if i % i_image == 0:
            h, w = test_data['target_gt'].shape[2:]

            albedo_all = torch.zeros(3, h, 2 * w)
            albedo_all[:, :, :w] = ret['albedo_gt']
            albedo_all[:, :, w:] = ret['albedo']
            albedo_all = albedo_all ** (1.0 / 2.2)

            normal_all = torch.zeros(3, h, 2 * w)
            normal_all[:, :, :w] = ret['normal_gt']
            normal_all[:, :, w:] = ret['normal']
            normal_all = 0.5 * (normal_all + 1)

            rough_all = torch.zeros(1, h, 2 * w)
            rough_all[:, :, :w] = ret['roughness_gt']
            rough_all[:, :, w:] = ret['roughness']
            rough_all = 0.5 * (rough_all + 1)

            saveimg_from_torch(f'{out_dir}albedo_{i}.png', albedo_all, iscolor=True)
            saveimg_from_torch(f'{out_dir}normal_{i}.png', normal_all, iscolor=False)
            saveimg_from_torch(f'{out_dir}rough_{i}.png', rough_all, iscolor=False)
            # # write the pred/gt rgb images and depths
            # writer.add_image(prefix + 'albedo', albedo_all, i)
            # writer.add_image(prefix + 'normal', normal_all, i)
            # writer.add_image(prefix + 'roughness', rough_all, i)

    writer.add_scalar('test/psnr_albedo', psnr_albedo / len(test_loader), 0)
    writer.add_scalar('test/psnr_normal', psnr_normal / len(test_loader), 0)
    writer.add_scalar('test/psnr_rough', psnr_rough / len(test_loader), 0)


if __name__ == '__main__':
    main()
