from utils import *
from tqdm import tqdm
import time
import datetime
from termcolor import colored
import sys


colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'grey', 'white']


def print_state(name, start_time, max_it, global_step, epoch, gpu):
    elapsed_sec = time.time() - start_time
    elapsed_time = str(datetime.timedelta(seconds=elapsed_sec)).split('.')[0]
    remain_sec = elapsed_sec * max_it / global_step
    remain_time = str(datetime.timedelta(seconds=remain_sec)).split('.')[0]
    prog = global_step / max_it * 100

    print(colored(f'{name} epoch : {epoch} step: {global_step} / {max_it}, {prog:.1f}% elapsed:{elapsed_time}, remain:{remain_time}', colors[gpu]))




def record_image(writer, epoch, global_step, pred, gt):
    h, w = pred.shape[2:]
    normal_all = torch.zeros(3, h, 2 * w)
    normal_all[:, :, :w] = (0.5 * (gt + 1))[0]
    normal_all[:, :, w:] = (0.5 * (pred + 1))[0]
    writer.add_image(f'normal_{epoch}', normal_all, global_step)


def eval_model(writer, model, val_loader, gpu, epoch):
    model.switch_to_eval()

    with torch.no_grad():
        mse_normal = 0.0
        angerr_normal = 0.0
        for val_data in tqdm(val_loader):
            rgbd = val_data['rgbd'].cuda(gpu)
            seg = val_data['seg'].cuda(gpu)
            normal_gt = val_data['normal'].cuda(gpu)
            light = val_data['light'].cuda(gpu)
            segAll = seg[:, 1:, ...].cuda(gpu)
            normal_pred = model.normal_net(rgbd)
            mse_normal += img2mse(normal_pred, normal_gt, segAll)
            angerr_normal += img2angerr(normal_pred, normal_gt, segAll)

        mse_normal = mse_normal.cpu().numpy() / len(val_loader)
        writer.add_scalar('mse_normal', mse_normal, epoch)
        psnr_normal = mse2psnr(mse_normal)
        writer.add_scalar('psnr_normal', psnr_normal, epoch)
        angerr_normal = angerr_normal.cpu().numpy() / len(val_loader)
        angerr_normal = np.rad2deg(angerr_normal)
        writer.add_scalar('angerr_normal', angerr_normal, epoch)
    model.switch_to_train()
