import torch.multiprocessing

from record import print_state, record_images, eval_model, quality_eval_model
from forward import model_forward
import time
import random
from utils import *
from loader import load_id_wandb, load_dataloader, load_model
import os

# root = '/new_disk2/happily/Data'
# root = '/media/vig-titan-103/mybookduo'
root = '/home/happily/Data'

def train(gpu, num_gpu, config, debug=False, phase='TRAIN', is_DDP=False, resume=False, run_id=None):
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2918', world_size=num_gpu, rank=gpu)
        torch.cuda.set_device(gpu)
    else:
        num_gpu = 1
    record_flag = ((is_DDP and gpu == 0) or not is_DDP) and not debug
    stage, cfg, model_type, run_id, wandb_obj, dataRoot, outputRoot, experiment = load_id_wandb(config, record_flag, resume, root, run_id)

    ckpt_prefix = osp.join(experiment, f'model_{model_type}')
    exp_name = osp.basename(experiment)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = torch.cuda.amp.GradScaler()
    curr_model, helper_dict = load_model(stage, cfg, gpu, experiment, phase=phase, is_DDP=is_DDP, wandb_obj=wandb_obj)
    train_loader, val_loader, train_sampler = load_dataloader(stage, dataRoot, cfg, debug, is_DDP, num_gpu, record_flag)

    global_step = curr_model.start_step

    scalars_to_log = {}
    max_iterations = len(train_loader) * cfg.nepoch
    start_time = time.time()
    epoch = global_step // len(train_loader)

    i_img = cfg.i_img
    while global_step < max_iterations:
        if not train_sampler is None:
            train_sampler.set_epoch(epoch)
        for train_data in train_loader:
            global_step += 1
            save_image_flag = global_step % i_img == 1
            train_data = tocuda(train_data, gpu, cfg.pinned)
            total_loss, pred = model_forward(stage, phase, curr_model, helper_dict, train_data, cfg, scalars_to_log, save_image_flag)

            curr_model.optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(curr_model.optimizer)
            scaler.update()
            curr_model.scheduler.step()

            if record_flag:
                if global_step % cfg.i_print == 0:
                    scalars_to_log['lr'] = curr_model.scheduler.get_last_lr()[0]
                    wandb_obj.log(scalars_to_log, step=global_step)
                    print_state(exp_name, start_time, max_iterations, global_step, gpu)
                if save_image_flag:
                    record_images(stage, cfg, wandb_obj, train_data, pred, global_step)
                    i_img = int(i_img * cfg.i_img_up)

                if global_step % cfg.i_save == 0:
                    fpath = f'{ckpt_prefix}_{global_step:06d}.pth'
                    curr_model.save_model(fpath)
                    eval_model(stage, wandb_obj, curr_model, helper_dict, val_loader, gpu, cfg, global_step)
                    torch.cuda.empty_cache()

        epoch += 1
    if record_flag:
        fpath = f'{ckpt_prefix}_latest.pth'
        curr_model.save_model(fpath)
        eval_model(stage, wandb_obj, curr_model, helper_dict, val_loader, gpu, cfg, global_step)
        quality_eval_model(stage, wandb_obj, curr_model, helper_dict, val_loader, gpu, cfg)
        wandb_obj.finish()
    if is_DDP:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    debug = True
    phase = 'TRAIN'
    is_DDP = False
    resume = False
    train(gpu=0, num_gpu=1, debug=debug, phase='TRAIN', config='stage2_titan.yml', is_DDP=is_DDP, resume=resume, run_id='06271610_stage2')