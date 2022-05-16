import sys
import torch
import importlib

if __name__ == '__main__':
    if len(sys.argv) < 3:
        num_gpu = 2
        config = 'stage1-1_2.yml'
    else:
        num_gpu = int(sys.argv[1])
        config = sys.argv[2]

    debug = False
    phase = 'TRAIN'
    is_DDP = True

    train_module = importlib.import_module('train_stage_' + config.split('stage')[1].split('_')[0])
    train = train_module.train
    torch.multiprocessing.spawn(train, nprocs=num_gpu, args=(num_gpu, config, debug, phase, is_DDP))
