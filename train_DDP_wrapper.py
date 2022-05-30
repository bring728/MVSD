import sys
import torch
from train_stage import train
from output_stage import output_stage_func
import os

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    if len(sys.argv) < 3:
        num_gpu = 7
        config = 'stage2_0.yml'
    else:
        num_gpu = int(sys.argv[1])
        config = sys.argv[2]

    if num_gpu == 7:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

    debug = False
    phase = 'TRAIN'
    is_DDP = True
    resume = False
    torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, config, debug, phase, is_DDP, resume))
    print('training is done.')