import os
from tqdm import tqdm
from render_func import *
import time
import numpy as np
from mvsd_dataset import Openrooms_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
import random
import sys
from cfgnode import CfgNode
import datetime
from models import MVSDModel
from utils import *

dataRoot = '/home/happily/Data/OpenRoomsDataset/data/rendering/data_FF_10_640/'
output = '/home/happily/Data/output/MVSD/'
# dataRoot = 'D:/OpenRoomsDataset/data/rendering/data_FF_10_640/'
# output = 'D:/MVSD_output/mpi-640'


if not torch.cuda.is_available():
    assert 'gpu isnt ready!'

# debug = True
debug = False


def main():
    if len(sys.argv) < 4:
        # gpus = '0'
        gpus = '0,1,2,3,4,5,6,7'
        num_gpu = len(gpus.split(','))
        config = 'initbrdf.yml'
        id = '3456'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # Set the GPUs 2 and 3 to use
    else:
        gpus = sys.argv[1]
        num_gpu = len(gpus.split(','))
        config = sys.argv[2]
        id = sys.argv[3]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus  # Set the GPUs 2 and 3 to use

    with open(os.getcwd() + '/config/' + config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    grouptype = cfg.grouptype
    normlayer = cfg.normlayer
    batchsize = cfg.batchsize

    # current_time = datetime.now()
    experiment = f'{output}/mpi_{normlayer}_{grouptype}_{batchsize}'
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
    num_workers = cfg.num_workers
    albedo_lambda = cfg.albedo_lambda
    normal_lambda = cfg.normal_lambda
    rough_lambda = cfg.rough_lambda

    experiment_name = experiment.split('/')[-1]

    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:' + id, world_size=ngpus_per_node, rank=gpu)
    torch.cuda.set_device(gpu)

    # create training dataset
    ####################################
    pinned = True
    _Openrooms_dataset = Openrooms_dataset(dataRoot, num_view_min=cfg.num_view_min, num_view_max=cfg.num_view_max, phase='TRAIN',
                                           debug=debug)
    _Openrooms_sampler = torch.utils.data.distributed.DistributedSampler(_Openrooms_dataset)
    openrooms_loader = torch.utils.data.DataLoader(_Openrooms_dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                                                   pin_memory=pinned, sampler=_Openrooms_sampler)

    val_dataset = Openrooms_dataset(dataRoot, num_view_min=cfg.num_view_min, num_view_max=cfg.num_view_max, phase='TEST', debug=debug)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create IBRNet model
    mvsd_model = MVSDModel(cfg, gpu, experiment)

    scalars_to_log = {}
    global_step = mvsd_model.start_step + 1
    max_iterations = len(openrooms_loader) * cfg.nepoch

    u, v = np.meshgrid(np.arange(cfg.imWidth), np.arange(cfg.imHeight))
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5  # add half pixel
    u_n = 2.0 * u / cfg.imWidth - 1
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    v_n = 2.0 * v / cfg.imHeight - 1
    pixels = np.stack((u, v, np.ones_like(u), u_n, v_n), axis=1)
    pixels = torch.from_numpy(pixels).cuda()

    rng = np.random.RandomState(234)
    epoch = 0

    writer = None
    if gpu == 0:
        writer = SummaryWriter(experiment)
        print('saving tensorboard files to {}'.format(experiment))
        # print(f'max iteration : {iterations}')

    # iterations = range(start_step, iterations)
    # for global_step in tqdm(iterations):
    start_time = time0 = time.time()
    while global_step < max_iterations:
        np.random.seed()
        for train_data in openrooms_loader:
            if cfg.distributed:
                _Openrooms_sampler.set_epoch(epoch)
            num_view = train_data['im_list'][0].shape[0]
            depth_list = train_data['depth_list'][0].cuda(gpu, non_blocking=pinned)
            int_list = train_data['int_list'][0].cuda(gpu, non_blocking=pinned)
            c2w_list = train_data['c2w_list'][0].cuda(gpu, non_blocking=pinned)
            im_list = train_data['im_list'][0].cuda(gpu, non_blocking=pinned)
            target_gt = train_data['target_gt'][0].cuda(gpu, non_blocking=pinned)

            # load training rays
            num_batch = int(1.0 * cfg.ray_batch * cfg.num_view_min / num_view)
            select_inds = rng.choice(cfg.imWidth * cfg.imHeight, size=(num_batch,), replace=False)
            pixel_batch = pixels[select_inds].cuda(gpu, non_blocking=pinned)

            pixel_normalized = pixel_batch[:, 3:][None, None]
            target_gt = F.grid_sample(target_gt[None], pixel_normalized, align_corners=False)[0, :, 0].transpose(0, 1)

            albedo_gt = target_gt[..., :3]
            normal_gt = target_gt[..., 3:6]
            rough_gt = target_gt[..., 6:7]
            depth_gt = target_gt[..., 7:8]
            segBRDF = target_gt[..., 8:9]
            segAll = target_gt[..., 9:]
            pixelObjNum = torch.sum(segBRDF).cpu().data.item() + 1.0
            pixelAllNum = torch.sum(segAll).cpu().data.item() + 1.0

            featmaps = mvsd_model.feature_net(im_list)

            rgb_feat_viewdir_err = compute_projection(pixel_batch, depth_gt, int_list, c2w_list, depth_list, im_list, featmaps)
            brdf = mvsd_model.brdf_net(rgb_feat_viewdir_err)

            albedo = 0.5 * (brdf[..., :3] + 1)
            albedo = LSregress(albedo * segBRDF, albedo_gt * segBRDF, albedo)

            x_orig = brdf[..., 3:6]
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1)).expand_as(x_orig)
            normal = x_orig / torch.clamp(norm, min=1e-6)

            roughness = brdf[..., 6:]

            albedo_err = albedo_lambda * torch.sum((albedo - albedo_gt) * (albedo - albedo_gt) * segBRDF) / pixelObjNum / 3.0
            normal_err = normal_lambda * torch.sum((normal - normal_gt) * (normal - normal_gt) * segAll) / pixelAllNum / 3.0
            rough_err = rough_lambda * torch.sum((roughness - rough_gt) * (roughness - rough_gt) * segBRDF) / pixelObjNum

            total_err = albedo_err + normal_err + rough_err
            # compute loss
            mvsd_model.optimizer.zero_grad()
            total_err.backward()
            scalars_to_log['albedo_err'] = albedo_err.item()
            scalars_to_log['normal_err'] = normal_err.item()
            scalars_to_log['rough_err'] = rough_err.item()
            scalars_to_log['total_err'] = total_err.item()
            mvsd_model.optimizer.step()
            mvsd_model.scheduler.step()

            scalars_to_log['lr'] = mvsd_model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            # dt = time.time() - time0

            # Rest is logging
            if gpu == 0:
                for k in scalars_to_log.keys():
                    # logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                    writer.add_scalar(k, scalars_to_log[k], global_step)
                if global_step % cfg.i_print == 0:
                    # logstr = '{} Epoch: {}  step: {} '.format(experiment_name, epoch, global_step)

                    elapsed_sec = time.time() - start_time
                    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
                    remain_sec = elapsed_sec * max_iterations / global_step
                    remain_time = str(datetime.timedelta(seconds=remain_sec)).split('.')[0]
                    prog = global_step / max_iterations * 100

                    print(
                        f'P{experiment_name} epoch : {epoch} step: {global_step} / {max_iterations}, {prog:%.1f} elapsed:{elapsed_time}, remain:{remain_time}')
                    # print('each iter time {:.05f} seconds'.format(dt))

                if global_step % cfg.i_weights == 0:
                    fpath = os.path.join(experiment, 'model_{:06d}.pth'.format(global_step))
                    mvsd_model.save_model(fpath)

                if global_step % cfg.i_img == 0:
                    print('Logging a random validation view...')
                    val_data = next(val_loader_iterator)
                    log_view_to_tb(writer, global_step, cfg, gpu, mvsd_model, val_data, pixels, prefix='val/')
                    torch.cuda.empty_cache()
            global_step += 1
        epoch += 1

    if gpu == 0:
        fpath = os.path.join(experiment, 'model_latest.pth')
        mvsd_model.save_model(fpath)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
