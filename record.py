import torch

from utils import *
from forward import model_forward
from tqdm import tqdm
import time
import datetime
from termcolor import colored
import torch.nn.functional as F
import wandb

colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'grey', 'white']


def print_state(name, start_time, max_it, global_step, gpu):
    elapsed_sec = time.time() - start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
    remain_sec = elapsed_sec * max_it / global_step
    remain_time = str(datetime.timedelta(seconds=remain_sec)).split('.')[0]
    prog = global_step / max_it * 100

    print(colored(f'{name} - step: {global_step} / {max_it}, {prog:.1f}% elapsed:{elapsed_time}, total:{remain_time}',
                  colors[gpu]))


def record_images(stage, cfg, wandb_obj, data, pred, step, val=False):
    prefix = ''
    if val:
        prefix = 'val_media/'
    log_image_dict = {}

    if stage == '1':
        imgs = [data['input'][0, :3, ...] ** (1.0 / 2.2), (0.5 * (pred['normal'][0] + 1)),
                (0.5 * (data['normal_gt'][0] + 1))]
        num_img_all = 3
        c, h, w = imgs[0].shape
        image_input = torch.zeros(c, h, num_img_all * w)
        for i in range(num_img_all):
            image_input[:, :, i * w:(i + 1) * w] = imgs[i].type(torch.float32)

        imgs = []
        imgs.append(data['input'][0, 3, ...].expand(pred['normal'][0].size()))
        if 'envmaps_DL_gt' in data:
            env_pred = env_vis(pred['envmaps_DL'][0], cfg.DL.env_height, cfg.DL.env_width, cfg.DL.env_rows, cfg.DL.env_cols)
            imgs.append(F.interpolate(env_pred[None], size=[240, 320])[0])
            env_gt = env_vis(data['envmaps_DL_gt'][0], cfg.DL.env_height, cfg.DL.env_width, cfg.DL.env_rows, cfg.DL.env_cols)
            imgs.append(F.interpolate(env_gt[None], size=[240, 320])[0])
        else:
            imgs.append(torch.zeros_like(imgs[0]))
            imgs.append(torch.zeros_like(imgs[0]))

        c, h, w = imgs[0].shape
        image_output = torch.zeros(c, h, num_img_all * w)
        for i in range(num_img_all):
            image_output[:, :, i * w:(i + 1) * w] = imgs[i].type(torch.float32)

        image = torch.cat([image_input, image_output], dim=1)
        log_image_dict[prefix + 'directlight'] = wandb.Image(image)

    elif stage == '2':
        size_tmp = data['normal'][0].size()
        # depth = data['depth'][0, 0] / torch.quantile(data['depth'][0], 0.9)
        # depth = torch.clamp(depth, 0, 1)
        depth = data['depth_norm'][0]

        imgs = [data['rgb'][0, 0] ** (1.0 / 2.2), (0.5 * (data['normal'][0] + 1)), depth.expand(size_tmp),
                data['conf'][0].expand(size_tmp)]
        env = env_vis(data['envmaps_DL'][0], cfg.DL.env_height, cfg.DL.env_width, cfg.DL.env_rows, cfg.DL.env_cols)
        imgs.append(F.interpolate(env[None], size=[240, 320])[0])

        num_img = len(imgs)
        c, h, w = imgs[0].shape
        image_input = torch.zeros([c, h, num_img * w], dtype=torch.float32)
        for i in range(num_img):
            image_input[:, :, i * w:(i + 1) * w] = imgs[i].type(torch.float32)

        imgs = [(pred['albedo'][0]), data['albedo_gt'][0], pred['rough'][0].expand(size_tmp), data['rough_gt'][0].expand(size_tmp),
                pred['conf'][0].expand(size_tmp)]
        num_img = len(imgs)
        c, h, w = imgs[0].shape
        image_output = torch.zeros([c, h, num_img * w], dtype=torch.float32)
        for i in range(num_img):
            image_output[:, :, i * w:(i + 1) * w] = imgs[i].type(torch.float32)

        image = torch.cat([image_input, image_output], dim=1)
        log_image_dict[prefix + 'BRDF'] = wandb.Image(image)

    wandb_obj.log(log_image_dict, step=step)


def eval_model(stage, wandb_obj, model, helper, val_loader, gpu, cfg, step):
    model.switch_to_eval()
    scalars_to_log_all = {}
    scalars_to_log = {}
    with torch.no_grad():
        for i, val_data in enumerate(tqdm(val_loader)):
            val_data = tocuda(val_data, gpu, False)
            _ = model_forward(stage, 'TEST', model, helper, val_data, cfg, scalars_to_log, False)
            for k in scalars_to_log:
                val_k = k.replace('train', 'val')
                if val_k in scalars_to_log_all:
                    scalars_to_log_all[val_k] += scalars_to_log[k]
                else:
                    scalars_to_log_all[val_k] = 0.0

    for val_k in scalars_to_log_all:
        scalars_to_log_all[val_k] /= len(val_loader)
        if val_k == 'val/normal_ang_err':
            scalars_to_log_all[val_k] = np.rad2deg(scalars_to_log_all[val_k])

    wandb_obj.log(scalars_to_log_all, step=step)
    model.switch_to_train()
    return


def quality_eval_model(stage, wandb_obj, model, helper, val_loader, gpu, cfg):
    model.switch_to_eval()
    with torch.no_grad():
        scalars_to_log = {}
        for i, val_data in enumerate(tqdm(val_loader)):
            if i > 10:
                break
            val_data = tocuda(val_data, gpu, False)
            total_loss, pred = model_forward(stage, 'TEST', model, helper, val_data, cfg, scalars_to_log, True)
            record_images(stage, cfg, wandb_obj, val_data, pred, i, val=True)
    model.switch_to_train()
    return