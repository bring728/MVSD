from forward import model_forward
from utils import *
from loader import load_id_wandb, load_dataloader, load_model
from tqdm import tqdm
import threading

root = '/new_disk/happily/Data'


def output_stage_func(gpu, num_gpu, config, phase='ALL', debug=True, is_DDP=False, resume=False):
    if not debug or not resume or not phase == 'ALL':
        raise Exception('check..!')
    if is_DDP:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2958', world_size=num_gpu, rank=gpu)
        torch.cuda.set_device(gpu)
    else:
        num_gpu = 1

    record_flag = ((is_DDP and gpu == 0) or not is_DDP) and not debug
    stage, cfg, model_type, run_id, wandb_obj, dataRoot, outputRoot, experiment = load_id_wandb(config, record_flag, resume, root)
    if not stage == '1-2':
        raise Exception('check..222!')

    curr_model, helper_dict = load_model(stage, cfg, gpu, experiment, phase=phase, is_DDP=is_DDP, wandb_obj=wandb_obj)
    all_loader, _, _ = load_dataloader(stage, dataRoot, cfg, phase, debug, is_DDP, num_gpu, record_flag)

    scalars_to_log = {}
    for all_data in tqdm(all_loader):
        all_data = tocuda(all_data, gpu, cfg.pinned, not stage.startswith('1'))
        with torch.no_grad():
            total_loss, pred = model_forward(stage, phase, curr_model, helper_dict, all_data, cfg, scalars_to_log, False)
            name = all_data['name']
            normal_name = [n.format('normalest', 'h5') for n in name]
            dl_name = [n.format('DLest', 'h5') for n in name]
            normal_pred = pred['normal']
            dl_pred = pred['DL']
            threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(normal_name, normal_pred)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            threads = [threading.Thread(target=th_save_h5, args=(name, img,)) for name, img in zip(dl_name, dl_pred)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()



    if is_DDP:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    debug = True
    phase = 'ALL'
    is_DDP = False
    resume = True
    output_stage_func(gpu=0, num_gpu=1, debug=debug, phase='ALL', config='stage1-2_1.yml', is_DDP=is_DDP, resume=resume)
