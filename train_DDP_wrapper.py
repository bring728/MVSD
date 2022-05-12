import sys
from train_stage_1_2 import train
import torch

if not torch.cuda.is_available():
    assert 'gpu isnt ready!'

if __name__ == '__main__':
    if len(sys.argv) < 3:
        num_gpu = 2
        config = 'stage1-2_0.yml'
    else:
        num_gpu = int(sys.argv[1])
        config = sys.argv[2]

    debug = False
    phase = 'TRAIN'
    is_DDP = True
    torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, config, debug, phase, is_DDP))
