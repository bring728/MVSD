import sys
import torch
from stage_train import train
# from stage_preprocess import preprocess_stage_func

# import imageio
# imageio.plugins.freeimage.download()

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        num_gpu = 4
        config = 'stage2_0.yml'
    else:
        num_gpu = int(sys.argv[1])
        config = sys.argv[2]

    is_DDP = True
    debug = False
    phase = 'TRAIN'
    resume = False
    torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, config, debug, phase, is_DDP, resume))
    # resume = True
    # torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, config, debug, phase, is_DDP, resume, '06232156_stage2'))
    # debug = True
    # resume = True
    # torch.multiprocessing.spawn(output_stage_func, nprocs=num_gpu, args=(num_gpu, config, debug, True, resume, '06221801_stage1-2'))
    # torch.multiprocessing.spawn(preprocess_stage_func, nprocs=num_gpu, args=(num_gpu, True))
    print('training is done.')