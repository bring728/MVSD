from utils import *
from tqdm import tqdm
import time
import datetime
from termcolor import colored
import sys
import torch.nn.functional as F



colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'grey', 'white']


def print_state(name, start_time, max_it, global_step, epoch, gpu):
    elapsed_sec = time.time() - start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
    remain_sec = elapsed_sec * max_it / global_step
    remain_time = str(datetime.timedelta(seconds=remain_sec)).split('.')[0]
    prog = global_step / max_it * 100

    print(colored(f'{name} epoch : {epoch} step: {global_step} / {max_it}, {prog:.1f}% elapsed:{elapsed_time}, remain:{remain_time}', colors[gpu]))

def record_images(writer, global_step, imgs, prefix):
    num_img = len(imgs)
    c, h, w = imgs[0].shape
    image = torch.zeros(c, h, num_img * w)
    for i in range(num_img):
        image[:, :, i * w:(i+1) * w] = imgs[i]
    writer.add_image(f'{prefix}', image, global_step)


def eval_model(writer, model, helper, val_loader, gpu, epoch, cfg, type, save):
    model.switch_to_eval()

    with torch.no_grad():
        depth_L1_loss = 0.0
        depth_conf_loss = 0.0
        depth_gaussian_loss = 0.0
        err_diff = 0.0
        normal_mse_err = 0.0
        normal_ang_err = 0.0
        save_period = 5
        count=0
        for i, val_data in enumerate(tqdm(val_loader)):
            val_data = tocuda(val_data, gpu)
            if type == 'depth':
                depthGT = val_data['depthGT']
                segAll = val_data['seg'][:, 1:, ...]
                before_err = torch.mean(val_data['depth_err'])

                depthpred, conf_or_var = model.depth_refine_net(val_data['input'])
                if cfg.loss == 'conf':
                    L1loss = torch.sum(torch.abs(depthpred - depthGT) * conf_or_var * segAll) / (torch.sum(segAll) + TINY_NUMBER)
                    confloss = torch.mean(1 - conf_or_var)
                    depth_L1_loss += L1loss.item()
                    depth_conf_loss += confloss.item()

                elif cfg.loss == 'gaussian':
                    L1loss = torch.sum(torch.abs(depthpred - depthGT) * segAll) / (torch.sum(segAll) + TINY_NUMBER)
                    gaussian_loss = helper(depthGT, depthpred, conf_or_var)
                    depth_L1_loss += L1loss.item()
                    depth_gaussian_loss += gaussian_loss.item()

                depth_input = cvimg_from_torch(val_data['input'][0, 3:4, ...], False)
                depth_pred = cvimg_from_torch(depthGT[0], False)
                depth_GT = cvimg_from_torch(depthpred[0], False)
                cv2.hconcat([depth_input, depth_pred, depth_GT])

                err_diff += (before_err - L1loss).item()

            elif type == 'normal':
                normal_pred = model.normal_net(val_data['input'])
                normal_gt = val_data['normal']
                segAll = val_data['seg'][:, 1:, ...]
                normal_mse_err += img2mse(normal_pred, normal_gt, segAll).item()
                normal_ang_err += img2angerr(normal_pred, normal_gt, segAll).item()
                if save and i % save_period == 0:
                    count+=1
                    if count > 5:
                        break
                    depth_input = val_data['input'][0, 3, ...]
                    depth_input = depth_input.expand(normal_gt[0].size())
                    imgs = [(0.5 * (normal_gt[0] + 1)), (0.5 * (normal_pred[0] + 1)), depth_input]
                    record_images(writer, i, imgs, 'normal_last')
                        # cv2.imwrite(f'{count}_with_gt.png', cvimg_from_torch(normal_pred[0] + 1 / 2, False))
                        # cv2.imwrite(f'{count}_gt.png', cvimg_from_torch(normal_gt[0] + 1 / 2, False))

            # elif cfg.model_type == 'light':
            #     normal_pred = model.normal_net(rgbd)
            #     rgbdn = torch.cat([rgbd, normal_pred], dim=1)
            #     axis, sharpness, intensity = model.DL_net(rgbdn)
            #     segBRDF = F.adaptive_avg_pool2d(segBRDF, (cfg.env_rows, cfg.env_cols))
            #     notDarkEnv = (torch.mean(torch.mean(torch.mean(envmaps_gt, 4), 4), 1, True) > 0.001).float()
            #     segEnvBatch = (segBRDF * envmapsIndBatch.expand_as(segBRDF)).unsqueeze(-1).unsqueeze(-1)
            #     segEnvBatch = (segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)).expand_as(envmaps_gt)
            #
            #     envmap_pred = sg2env.forward(axis, sharpness, intensity)
            #     if cfg.scale_inv:
            #         envmap_pred = LSregress(envmap_pred.detach() * segEnvBatch, envmaps_gt * segEnvBatch, envmap_pred)
            #     env_err += img2log_mse(envmap_pred, envmaps_gt, segEnvBatch)

    if not save:
        if type == 'depth':
            depth_L1_loss = depth_L1_loss / len(val_loader)
            writer.add_scalar('val/depth_L1_loss', depth_L1_loss, epoch)
            err_diff = err_diff / len(val_loader)
            writer.add_scalar('val/err_diff', err_diff, epoch)
            depth_conf_loss = depth_conf_loss / len(val_loader)
            writer.add_scalar('val/depth_conf_loss', depth_conf_loss, epoch)
            depth_gaussian_loss = depth_gaussian_loss / len(val_loader)
            writer.add_scalar('val/depth_gaussian_loss', depth_gaussian_loss, epoch)

        elif type == 'normal':
            mse_normal = normal_mse_err / len(val_loader)
            writer.add_scalar('val/mse_normal', mse_normal, epoch)
            # psnr_normal = mse2psnr(mse_normal)
            # writer.add_scalar('psnr_normal', psnr_normal, epoch)
            angerr_normal = normal_ang_err / len(val_loader)
            angerr_normal = np.rad2deg(angerr_normal)
            writer.add_scalar('val/angerr_normal', angerr_normal, epoch)
            print(mse_normal, angerr_normal)
        # elif type == 'light':
        #     env_err = env_err.cpu().numpy() / len(val_loader)
        #     writer.add_scalar('logmse_envmap', env_err, epoch)
    model.switch_to_train()
    return

