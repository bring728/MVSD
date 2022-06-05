from forward import model_forward
from utils import *
from loader import load_id_wandb, load_dataloader, load_model
from tqdm import tqdm
import threading

root = '/new_disk/happily/Data'


def output_stage_func(gpu, num_gpu, config, debug=True, is_DDP=False, resume=False):
    if not debug or not resume:
        raise Exception('check..!')
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2958', world_size=num_gpu, rank=gpu)
        torch.cuda.set_device(gpu)
    else:
        num_gpu = 1

    record_flag = False
    stage, cfg, model_type, run_id, wandb_obj, dataRoot, outputRoot, experiment = load_id_wandb(config, record_flag, resume, root)

    curr_model, helper_dict = load_model(stage, cfg, gpu, experiment, phase='TEST', is_DDP=is_DDP, wandb_obj=wandb_obj)
    train_loader, val_loader, train_sampler = load_dataloader(stage, dataRoot, cfg, debug, is_DDP, gpu, num_gpu, record_flag)

    scalars_to_log = {}
    for train_data in tqdm(train_loader):
        train_data = tocuda(train_data, gpu, cfg.pinned)
        with torch.no_grad():
            total_loss, pred = model_forward(stage, curr_model, 'output', helper_dict, train_data, cfg, scalars_to_log, False)
            name = train_data['name']
            normal_name = [n.format('normalest', 'h5') for n in name]
            normal_pred = pred['normal']
            threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(normal_name, normal_pred)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            if 'DL' in pred:
                dl_name = [n.format('DLest', 'h5') for n in name]
                dl_pred = pred['DL']
                threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(dl_name, dl_pred)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

    if gpu == 0:
        for val_data in tqdm(val_loader):
            val_data = tocuda(val_data, gpu, cfg.pinned)
            with torch.no_grad():
                total_loss, pred = model_forward(stage, curr_model, helper_dict, val_data, cfg, scalars_to_log, False)
                name = val_data['name']
                normal_name = [n.format('normalest', 'h5') for n in name]
                normal_pred = pred['normal']
                threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(normal_name, normal_pred)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                if 'DL' in pred:
                    dl_name = [n.format('DLest', 'h5') for n in name]
                    dl_pred = pred['DL']
                    threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(dl_name, dl_pred)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()

    if is_DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    output_stage_func(gpu=0, num_gpu=1, debug=True, config='stage1-2_1.yml', is_DDP=False, resume=True)
