import sys
import torch
from stage_train import train
# from stage_preprocess import preprocess_stage_func
from stage_output import output_stage_func

# import imageio
# imageio.plugins.freeimage.download()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        num_gpu = 4
        config = 'stage1-1_0.yml'
    else:
        num_gpu = int(sys.argv[1])
        config = sys.argv[2]

    is_DDP = True
    debug = False
    phase = 'TRAIN'
    resume = False
    torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, config, debug, phase, is_DDP, resume))

    # debug = True
    # resume = True
    # torch.multiprocessing.spawn(output_stage_func, nprocs=num_gpu, args=(num_gpu, config, debug, True, resume))
    # torch.multiprocessing.spawn(preprocess_stage_func, nprocs=num_gpu, args=(num_gpu, True))
    print('training is done.')