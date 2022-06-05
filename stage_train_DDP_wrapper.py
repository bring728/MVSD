import sys
import torch
from stage_train import train
# from stage_preprocess import preprocess_stage_func
# from stage_output import output_stage_func
import os

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        num_gpu = 8
        config = 'stage1-2_1.yml'
    else:
        num_gpu = int(sys.argv[1])
        config = sys.argv[2]

    if num_gpu == 7:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

    debug = False
    phase = 'TRAIN'
    resume = False
    torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, config, debug, phase, True, resume))
    # torch.multiprocessing.spawn(preprocess_stage_func, nprocs=num_gpu, args=(num_gpu, True))
    # torch.multiprocessing.spawn(output_stage_func, nprocs=num_gpu, args=(num_gpu, debug, config, True, True))
    print('training is done.')