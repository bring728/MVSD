from utils import *
from tqdm import tqdm
import time
import datetime
# from termcolor import colored
import sys
import torch.nn.functional as F

colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'grey', 'white']


def print_state(name, start_time, max_it, global_step, epoch, gpu):
    elapsed_sec = time.time() - start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
    remain_sec = elapsed_sec * max_it / global_step
    remain_time = str(datetime.timedelta(seconds=remain_sec)).split('.')[0]
    prog = global_step / max_it * 100

    print(f'{name} epoch : {epoch} step: {global_step} / {max_it}, {prog:.1f}% elapsed:{elapsed_time}, remain:{remain_time}')
    # print(colored(f'{name} epoch : {epoch} step: {global_step} / {max_it}, {prog:.1f}% elapsed:{elapsed_time}, remain:{remain_time}', colors[gpu]))


def record_images(writer, global_step, imgs, prefix):
    num_img = len(imgs)
    c, h, w = imgs[0].shape
    image = torch.zeros(c, h, num_img * w)
    for i in range(num_img):
        image[:, :, i * w:(i + 1) * w] = imgs[i]
    writer.add_image(f'{prefix}', image, global_step)


def eval_model(writer, model, helper, val_loader, gpu, epoch, cfg, model_type):
    model.switch_to_eval()
    with torch.no_grad():
        normal_mse_err = 0.0
        normal_ang_err = 0.0
        env_err = 0.0
        for i, val_data in enumerate(tqdm(val_loader)):
            val_data = tocuda(val_data, gpu)
            if model_type == 'normal':
                normal_pred = model.normal_net(val_data['input'])
                normal_gt = val_data['normal_gt']
                segAll = val_data['seg'][:, 1:, ...]
                normal_mse_err += img2mse(normal_pred, normal_gt, segAll).item()
                normal_ang_err += img2angerr(normal_pred, normal_gt, segAll).item()

            elif model_type == 'light':
                rgbdc = val_data['input']
                segBRDF = val_data['seg'][:, :1, ...]
                envmaps_gt = val_data['envmaps_gt']
                envmapsIndBatch = val_data['envmapsInd']

                normal_pred = model.normal_net(rgbdc)
                normal_pred = 0.5 * (normal_pred + 1)
                rgbdcn = torch.cat([rgbdc, normal_pred], dim=1)
                axis, sharpness, intensity = model.DL_net(rgbdcn)
                segBRDF = F.adaptive_avg_pool2d(segBRDF, (cfg.env_rows, cfg.env_cols))
                notDarkEnv = (torch.mean(torch.mean(torch.mean(envmaps_gt, 4), 4), 1, True) > 0.001).float()
                segEnvBatch = (segBRDF * envmapsIndBatch.expand_as(segBRDF)).unsqueeze(-1).unsqueeze(-1)
                segEnvBatch = (segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)).expand_as(envmaps_gt)

                envmap_pred = helper.forward(axis, sharpness, intensity)
                envmap_pred_scaled = LSregress(envmap_pred.detach() * segEnvBatch, envmaps_gt * segEnvBatch, envmap_pred)
                env_err += img2log_mse(envmap_pred_scaled, envmaps_gt, segEnvBatch).item()

    if model_type == 'normal':
        mse_normal = normal_mse_err / len(val_loader)
        writer.add_scalar('val/mse_normal', mse_normal, epoch)
        # psnr_normal = mse2psnr(mse_normal)
        # writer.add_scalar('psnr_normal', psnr_normal, epoch)
        angerr_normal = normal_ang_err / len(val_loader)
        angerr_normal = np.rad2deg(angerr_normal)
        writer.add_scalar('val/angerr_normal', angerr_normal, epoch)
        print(mse_normal, angerr_normal)
    elif model_type == 'light':
        env_err = env_err / len(val_loader)
        writer.add_scalar('val/logmse_envmap', env_err, epoch)
    model.switch_to_train()
    return


def quality_eval_model(writer, model, helper, val_loader, gpu, cfg, model_type):
    model.switch_to_eval()
    with torch.no_grad():
        for i, val_data in enumerate(tqdm(val_loader)):
            if i > 10:
                break
            val_data = tocuda(val_data, gpu)
            if model_type == 'normal':
                normal_pred = model.normal_net(val_data['input'])
                normal_gt = val_data['normal_gt']
                depth_input = val_data['input'][0, 3, ...]
                depth_input = depth_input.expand(normal_gt[0].size())
                imgs = [(0.5 * (normal_gt[0] + 1)), (0.5 * (normal_pred[0] + 1)), depth_input]
                record_images(writer, i, imgs, 'normal_last')

            elif model_type == 'light':
                rgbdc = val_data['input']
                envmaps_gt = val_data['envmaps']
                normal_pred = model.normal_net(rgbdc)
                normal_pred = 0.5 * (normal_pred + 1)
                rgbdcn = torch.cat([rgbdc, normal_pred], dim=1)
                axis, sharpness, intensity = model.DL_net(rgbdcn)
                envmap_pred = helper.forward(axis, sharpness, intensity)
                imgs = [env_vis(envmaps_gt[0], cfg.env_height, cfg.env_width, cfg.env_rows, cfg.env_cols),
                        env_vis(envmap_pred[0], cfg.env_height, cfg.env_width, cfg.env_rows, cfg.env_cols)]
                record_images(writer, i, imgs, 'light_last')

    model.switch_to_train()
    return

