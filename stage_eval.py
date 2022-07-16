from record import print_state, record_images, eval_model, quality_eval_model
from forward import model_forward
import time
import random
from utils import *
from loader import load_id_wandb, load_dataloader, load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

root = '/home/happily/Data'

def eval_func(gpu, num_gpu, config, resume=False, run_id=None):
    record_flag = True
    stage, cfg, model_type, run_id, wandb_obj, dataRoot, outputRoot, experiment = load_id_wandb(config, record_flag, resume, root, run_id, resume_eval=True)

    seed = cfg.randomseed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    curr_model, helper_dict = load_model(stage, cfg, gpu, experiment, phase='TEST', is_DDP=False, wandb_obj=wandb_obj)
    _, val_loader, _ = load_dataloader(stage, dataRoot, cfg, False, False, num_gpu, record_flag)

    experiment = experiment.replace('_eval2', '')
    ckpts = [os.path.join(experiment, f) for f in sorted(os.listdir(experiment)) if f.endswith('.pth')]

    for ckpt in ckpts:
        curr_model.load_model(ckpt)
        step = osp.basename(ckpt).split('_')[-1].split('.')[0]
        if step == 'latest':
            step = 200000
        else:
            step = int(step)
        eval_model(stage, wandb_obj, curr_model, helper_dict, val_loader, gpu, cfg, step)
        torch.cuda.empty_cache()
    wandb_obj.finish()

if __name__ == "__main__":
    eval_func(gpu=0, num_gpu=1, config='stage2_daniff.yml', run_id='07101614_stage2')